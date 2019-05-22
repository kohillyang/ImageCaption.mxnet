import json
import logging
import os
import time

import cv2
import gluoncv
import mxnet as mx
import mxnet.autograd as ag
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.rnn import LSTMCell
from mxnet.metric import TopKAccuracy, Loss, Accuracy
from nltk.translate.bleu_score import corpus_bleu
from dataset import CaptionDataSet, LeftTopPad
from parallel import DataParallelModel
from resnetv1b import resnet50_v1b
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "7,8"


class AdaptiveAttention(torch.nn.Module):
    def __init__(self, encoder_dim, attention_dim=2048):
        """
        An implementation of the AdaptiveAttention, see https://arxiv.org/pdf/1612.01887.pdf.
        :param attention_dim:
        """
        super(AdaptiveAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(attention_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.adaptive_w_h = nn.Linear(attention_dim, attention_dim)
        self.adaptive_w_x = nn.Linear(attention_dim, attention_dim)
        self.adaptive_w_s = nn.Linear(attention_dim, attention_dim)
        self.adaptive_w_s_1 = nn.Linear(attention_dim, encoder_dim)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor, x_t: torch.Tensor,
                memory_cell: torch.Tensor):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch_size, attention_dim)
        s_t = (self.adaptive_w_h(decoder_hidden) + self.adaptive_w_x(x_t)).sigmoid() * memory_cell.tanh()
        beta_pre = self.full_att((self.adaptive_w_s(s_t) + self.decoder_att(s_t)).tanh())

        att = self.full_att((att1 + att2).tanh()).squeeze(2)  # (batch_size, num_pixels)
        att_with_beta = torch.cat([att, beta_pre], dim=1)
        alpha = att_with_beta.softmax(dim=1)  # (batch_size, num_pixels)
        alpha_1 = alpha[:, 0:-1]
        beta = alpha[:, -1:]
        attention_weighted_encoding = (encoder_out * alpha_1.unsqueeze(2)).sum(dim=1)
        c_t = ((1 - beta) * attention_weighted_encoding) + self.adaptive_w_s_1((beta * s_t))
        return c_t, alpha_1


class DecoderWithAttention(nn.Module):
    def __init__(self, num_words, max_len=32, use_current_state=True, use_adaptive_attention=True):
        """
        :param num_words:
        :param max_len:
        :param use_current_state: whether to use current_state h_k to analyse where to look,
                see https://arxiv.org/pdf/1612.01887.pdf.
        """
        self.encoder_dim = 2048
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = 512
        self.lstm_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        if use_adaptive_attention:
            self.attention = AdaptiveAttention(attention_dim=self.hidden_size, encoder_dim=self.encoder_dim)
        else:
            assert False
        self.num_words = num_words

        self.init_h = nn.Linear(self.encoder_dim, self.hidden_size)
        self.init_c = nn.Linear(self.encoder_dim, self.hidden_size)
        self.f_beta = nn.Linear(self.hidden_size, 2048)
        self.out = nn.Linear(self.hidden_size + self.encoder_dim, self.num_words)

        self.embedding = nn.Embedding(num_words, self.hidden_size)
        self.dropout = nn.Dropout(p=.5)

        self.max_len = max_len
        self._use_current_state = use_current_state
        self._use_adaptive_attention = use_adaptive_attention

    def forward(self, encoder_output: torch.Tensor, label=None, max_length=None):
        no_label = label is None or max_length is None

        encoder_output = encoder_output.permute(0, 2, 3, 1)
        encoder_output = encoder_output.view(encoder_output.shape[0], -1, encoder_output.shape[3])
        batch_max_len = self.max_len if no_label else max_length

        # Initialize hidden states
        encoder_output_mean = encoder_output.mean(dim=1)
        h = self.init_h(encoder_output_mean)
        c = self.init_c(encoder_output_mean)

        # Two tensors to store outputs
        predictions = []
        alphas = []

        if not no_label:
            label_embedded = self.embedding(label)
        else:
            bs = encoder_output.shape[0]
            x_t = self.embedding(torch.zeros(bs, ).long().cuda())
        for t in range(batch_max_len):
            if not no_label:
                x_t = label_embedded[:, t]
            h, c = self.lstm_cell(x_t, [h, c])
            atten_weights, alpha = self.attention(encoder_output, h, x_t, c)
            atten_weights = self.f_beta(h).sigmoid() * atten_weights
            inputs = torch.cat([atten_weights, h], dim=1)
            preds = self.out(self.dropout(inputs))
            x_t = self.embedding(preds.argmax(dim=1))
            predictions.append(preds)
            alphas.append(alpha)
        predictions = torch.cat([x.unsqueeze(1) for x in predictions], dim=1)
        alphas = torch.cat([x.unsqueeze(1) for x in alphas], dim=1)

        return predictions, alphas


class EncoderDecoder(nn.Module):
    def __init__(self, num_words, test_max_len):
        super(EncoderDecoder, self).__init__()
        self.decoder = DecoderWithAttention(num_words=num_words, max_len=test_max_len)

    def forward(self, image, label, label_len):
        x = self.decoder(image, label, label_len)
        return x


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    def forward(self, predictions, labels, label_lengths):
        label_lengths = label_lengths.data.cpu().numpy().squeeze().tolist()
        losses = []
        for p, l, length in zip(predictions, labels, label_lengths):
            p = p[:(length - 1)]
            l = l[1:length]
            loss = self.criterion(p, l).sum()
            losses.append(loss)
        return sum(losses[1:], losses[0])


class BleuMetric(object):
    def __init__(self, name="blue", pred_index2words=None, label_index2words=None):
        self.preds = []
        self.labels = []
        self.name = name
        self.pred_index2words = pred_index2words
        self.label_index2words = label_index2words

    def update(self, label, pred):
        if isinstance(label, mx.nd.NDArray):
            label = label.asnumpy()
            pred = pred.asnumpy()
        label = label.astype(np.int)
        pred = pred.argmax(axis=1)
        label = [self.label_index2words[x] for x in label]
        pred = map(lambda x: self.pred_index2words[x], pred)
        pred = list(filter(lambda x: x != "<PAD>", pred))
        try:
            pred_len = pred.index("<END>")
        except ValueError:
            pred_len = len(pred)
        self.preds.append(pred[:pred_len])
        self.labels.append([label])

    def get(self):
        return self.name, corpus_bleu(self.labels, self.preds)

    def reset(self):
        self.preds = []
        self.labels = []


def validate(net, val_loader, gpu_id, train_index2words, val_index2words):
    metric = BleuMetric(pred_index2words=train_index2words, label_index2words=val_index2words)
    metruc_acc = Accuracy()
    metruc_acc.reset()
    metric.reset()
    for batch in tqdm.tqdm(val_loader):
        batch = [Variable(torch.from_numpy(x.asnumpy()).cuda()) for x in batch]
        image, label, label_len = batch
        label = label.long()
        label_len = label_len.long()
        predictions, alphas = net(image, None, None)
        for n, l in enumerate(label_len):
            l = int(l.data.cpu().numpy().squeeze().tolist())
            la = label[n, 1:l].data.cpu().numpy()
            pred = predictions[n, :].data.cpu().numpy()
            metric.update(la, pred)
            metruc_acc.update(mx.nd.array(la), mx.nd.array(predictions[n, :(l - 1)].data.cpu().numpy()))
    return metric.get()[1], metruc_acc.get()[1]


def main():
    epoches = 32
    gpu_id = 7
    ctx_list = [mx.gpu(x) for x in [7, 8]]
    log_interval = 100
    batch_size = 32
    start_epoch = 0
    # trainer_resume = resume + ".states" if resume is not None else None
    trainer_resume = None

    resume = None
    from mxnet.gluon.data.vision import transforms
    transform_fn = transforms.Compose([
        LeftTopPad(dest_shape=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset = CaptionDataSet(image_root="/data3/zyx/yks/coco2017/train2017",
                             annotation_path="/data3/zyx/yks/coco2017/annotations/captions_train2017.json",
                             transforms=transform_fn,
                             feature_hdf5="output/train2017.h5"
                             )
    val_dataset = CaptionDataSet(image_root="/data3/zyx/yks/coco2017/val2017",
                                 annotation_path="/data3/zyx/yks/coco2017/annotations/captions_val2017.json",
                                 words2index=dataset.words2index,
                                 index2words=dataset.index2words,
                                 transforms=transform_fn,
                                 feature_hdf5="output/val2017.h5"
                                 )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                            last_batch="discard")
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    num_words = dataset.words_count

    # set up logger
    save_prefix = "output/res50_"
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    net = EncoderDecoder(num_words=num_words, test_max_len=val_dataset.max_len).cuda()
    for name, p in net.named_parameters():
        if "bias" in name:
            p.data.zero_()
        else:
            p.data.normal_(0, 0.01)
        print(name)
    net = torch.nn.DataParallel(net)
    if resume is not None:
        net.collect_params().load(resume, allow_missing=True, ignore_extra=True)
        logger.info("Resumed form checkpoint {}.".format(resume))

    trainer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()),
                               lr=4e-4)
    criterion = Criterion()
    accu_top3_metric = TopKAccuracy(top_k=3)
    accu_top1_metric = Accuracy(name="batch_accu")
    ctc_loss_metric = Loss(name="ctc_loss")
    alpha_metric = Loss(name="alpha_loss")
    batch_bleu = BleuMetric(name="batch_bleu", pred_index2words=dataset.index2words,
                            label_index2words=dataset.index2words)
    epoch_bleu = BleuMetric(name="epoch_bleu", pred_index2words=dataset.index2words,
                            label_index2words=dataset.index2words)
    btic = time.time()
    logger.info(batch_size)
    logger.info(num_words)
    logger.info(len(dataset.words2index))
    logger.info(len(dataset.index2words))
    logger.info(dataset.words2index["<PAD>"])
    logger.info(val_dataset.words2index["<PAD>"])
    logger.info(len(val_dataset.words2index))
    for nepoch in range(start_epoch, epoches):
        if nepoch > 15:
            trainer.set_learning_rate(4e-5)
        logger.info("Current lr: {}".format(trainer.param_groups[0]["lr"]))
        accu_top1_metric.reset()
        accu_top3_metric.reset()
        ctc_loss_metric.reset()
        alpha_metric.reset()
        epoch_bleu.reset()
        batch_bleu.reset()
        for nbatch, batch in enumerate(tqdm.tqdm(dataloader)):
            batch = [Variable(torch.from_numpy(x.asnumpy()).cuda()) for x in batch]
            data, label, label_len = batch
            label = label.long()
            label_len = label_len.long()
            max_len = label_len.max().data.cpu().numpy()
            net.train()
            outputs = net(data, label, max_len)
            predictions, alphas = outputs
            ctc_loss = criterion(predictions, label, label_len)
            loss2 = 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()
            ((ctc_loss + loss2) / batch_size).backward()
            for group in trainer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-5, 5)

            trainer.step()
            if nbatch % 10 == 0:
                for n, l in enumerate(label_len):
                    l = int(l.data.cpu().numpy())
                    la = label[n, 1:l].data.cpu().numpy()
                    pred = predictions[n, :(l - 1)].data.cpu().numpy()
                    accu_top3_metric.update(mx.nd.array(la), mx.nd.array(pred))
                    accu_top1_metric.update(mx.nd.array(la), mx.nd.array(pred))
                    epoch_bleu.update(la, predictions[n, :].data.cpu().numpy())
                    batch_bleu.update(la, predictions[n, :].data.cpu().numpy())
                ctc_loss_metric.update(None, preds=mx.nd.array([ctc_loss.data.cpu().numpy()]) / batch_size)
                alpha_metric.update(None, preds=mx.nd.array([loss2.data.cpu().numpy()]))
                if nbatch % log_interval == 0 and nbatch > 0:
                    msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in [
                        epoch_bleu, batch_bleu, accu_top1_metric, accu_top3_metric, ctc_loss_metric, alpha_metric]])
                    logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                        nepoch, nbatch, log_interval * batch_size / (time.time() - btic), msg))
                    btic = time.time()
                    batch_bleu.reset()
                    accu_top1_metric.reset()
                    accu_top3_metric.reset()
                    ctc_loss_metric.reset()
                    alpha_metric.reset()
        net.eval()
        bleu, acc_top1 = validate(net, gpu_id=gpu_id,
                                  val_loader=val_loader,
                                  train_index2words=dataset.index2words,
                                  val_index2words=val_dataset.index2words)
        save_path = save_prefix + "_weights-%d-bleu-%.4f-%.4f.params" % (nepoch, bleu, acc_top1)
        torch.save(net.module.state_dict(), save_path)
        torch.save(trainer.state_dict(), save_path + ".states")
        logger.info("Saved checkpoint to {}.".format(save_path))


if __name__ == '__main__':
    main()
    # demo()
