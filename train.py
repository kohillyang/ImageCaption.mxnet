import json
import logging
import os
import time

import cv2
import gluoncv
import mxnet as mx
import mxnet.autograd as ag
import mxnet.gluon.nn as nn
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.rnn import LSTMCell
from mxnet.metric import TopKAccuracy, Loss, Accuracy
from nltk.translate.bleu_score import corpus_bleu
from dataset import CaptionDataSet, LeftTopPad
from parallel import DataParallelModel
from resnetv1b import resnet50_v1b


class Encoder(nn.HybridBlock):
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature = resnet50_v1b(dilated=False, pretrained=True)
        # self.feature.fc.weight.grad_req = "null"
        # self.feature.fc.bias.grad_req = "null"

    def hybrid_forward(self, F, x):
        feat = self.feature
        fm0 = feat.conv1(x)
        fm0 = feat.bn1(fm0)
        fm0 = feat.relu(fm0)
        fm0 = feat.maxpool(fm0)

        fm1 = feat.layer1(fm0)
        fm2 = feat.layer2(fm1)
        fm3 = feat.layer3(fm2)
        fm4 = feat.layer4(fm3)

        return fm4


class Attention(nn.HybridBlock):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Dense(attention_dim, flatten=False)  # linear layer to transform encoded image
        self.decoder_att = nn.Dense(attention_dim, flatten=False)  # linear layer to transform decoder's output
        self.full_att = nn.Dense(1, flatten=False,
                                 prefix="full_attention")  # linear layer to calculate values to be softmax-ed

    def hybrid_forward(self, F, encoder_out: nd.NDArray, decoder_hidden: nd.NDArray):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).expand_dims(axis=1)  # (batch_size, attention_dim)

        att = self.full_att(F.broadcast_add(att1, att2).tanh()).squeeze(axis=2)  # (batch_size, num_pixels)
        alpha = att.softmax(axis=1)  # (batch_size, num_pixels)
        attention_weighted_encoding = (F.broadcast_mul(encoder_out, alpha.expand_dims(2))).sum(
            axis=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Block):
    def __init__(self, num_words, max_len=185, use_current_state=True):
        """
        :param num_words:
        :param max_len:
        :param use_current_state: whether to use current_state h_k to analyse where to look,
                see https://arxiv.org/pdf/1612.01887.pdf.
        """
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = 512
        self.lstm_cell = LSTMCell(hidden_size=self.hidden_size)
        self.attention = Attention(attention_dim=self.hidden_size)
        self.num_words = num_words

        self.init_h = nn.Dense(self.hidden_size, flatten=False)
        self.init_c = nn.Dense(self.hidden_size, flatten=False)
        self.f_beta = nn.Dense(2048, flatten=False)
        self.out = nn.Dense(self.num_words, flatten=False)

        self.embedding = nn.Embedding(input_dim=num_words, output_dim=self.hidden_size)
        self.dropout = nn.Dropout(rate=.5)

        self.max_len = max_len
        self._use_current_state = use_current_state

    def forward(self, encoder_output: nd.NDArray, label=None, label_lengths=None):
        if label is None or label_lengths is None:
            return self.forward_test(encoder_output)

        encoder_output = nd.transpose(encoder_output, (0, 2, 3, 1))
        encoder_output = encoder_output.reshape((encoder_output.shape[0], -1, encoder_output.shape[3]))
        batch_max_len = int(label_lengths.max().asscalar()) - 1

        # Initialize hidden states
        encoder_output_mean = encoder_output.mean(axis=1)
        h = self.init_h(encoder_output_mean)
        c = self.init_c(encoder_output_mean)

        # Two tensors to store outputs
        predictions = []
        alphas = []

        label_embedded = self.embedding(label)

        for t in range(batch_max_len):
            if self._use_current_state:
                _, [h, c] = self.lstm_cell(label_embedded[:, t], [h, c])
                atten_weights, alpha = self.attention(encoder_output, h)
                atten_weights = self.f_beta(h).sigmoid() * atten_weights
                inputs = nd.concat(atten_weights, h, dim=1)
                preds = self.out(self.dropout(inputs))
                pass
            else:
                atten_weights, alpha = self.attention(encoder_output, h)
                atten_weights = nd.sigmoid(self.f_beta(h)) * atten_weights
                inputs = nd.concat(label_embedded[:, t], atten_weights, dim=1)
                _, [h, c] = self.lstm_cell(inputs, [h, c])
                preds = self.out(self.dropout(h))
            predictions.append(preds)
            alphas.append(alpha)
        predictions = nd.concat(*[x.expand_dims(axis=1) for x in predictions], dim=1)
        alphas = nd.concat(*[x.expand_dims(axis=1) for x in alphas], dim=1)

        return predictions, alphas

    def forward_test(self, encoder_output: nd.NDArray):
        bs = encoder_output.shape[0]

        encoder_output = nd.transpose(encoder_output, (0, 2, 3, 1))
        encoder_output = encoder_output.reshape((encoder_output.shape[0], -1, encoder_output.shape[3]))
        batch_max_len = self.max_len

        # Initialize hidden states
        encoder_output_mean = encoder_output.mean(axis=1)
        h = self.init_h(encoder_output_mean)
        c = self.init_c(encoder_output_mean)

        # Two tensors to store outputs
        predictions = []
        alphas = []

        last_preds = nd.zeros(shape=(bs,), ctx=encoder_output.context)

        for t in range(batch_max_len):
            atten_weights, alpha = self.attention(encoder_output, h)
            atten_weights = nd.sigmoid(self.f_beta(h)) * atten_weights
            last_preds_embedded = self.embedding(last_preds)
            inputs = nd.concat(last_preds_embedded, atten_weights, dim=1)
            _, [h, c] = self.lstm_cell(inputs, [h, c])
            preds = self.out(self.dropout(h))
            last_preds = preds.argmax(axis=1)
            predictions.append(preds)
            alphas.append(alpha)
        predictions = nd.concat(*[x.expand_dims(axis=1) for x in predictions], dim=1)
        alphas = nd.concat(*[x.expand_dims(axis=1) for x in alphas], dim=1)

        return predictions, alphas


class EncoderDecoder(nn.Block):
    def __init__(self, num_words, test_max_len):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        # self.encoder.hybridize()
        self.decoder = DecoderWithAttention(num_words=num_words, max_len=test_max_len)

    def forward(self, image, label, label_len):
        x = self.encoder(image)
        x = self.decoder(x, label, label_len)
        return x


class Criterion(nn.Block):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, predictions, labels, label_lengths):
        label_lengths = label_lengths.asnumpy().astype(int).tolist()
        losses = []
        for p, l, length in zip(predictions.log_softmax(axis=2), labels, label_lengths):
            p = p[:(length[0] - 1)]
            l = l[1:length[0]]
            loss = mx.nd.pick(data=p, index=l).sum()
            losses.extend(loss)
        return -1 * mx.nd.concat(*losses, dim=0)


class BleuMetric(object):
    def __init__(self, name="blue", pred_index2words=None, label_index2words=None):
        self.preds = []
        self.labels = []
        self.name = name
        self.pred_index2words = pred_index2words
        self.label_index2words = label_index2words

    def update(self, label, pred):
        label = label.asnumpy().astype(np.int)
        pred = pred.asnumpy().argmax(axis=1)

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
    for batch in val_loader:
        batch = [x.as_in_context(mx.gpu(gpu_id)) for x in batch]
        image, label, label_len = batch
        predictions, alphas = net(image, None, None)
        for n, l in enumerate(label_len):
            l = int(l.asscalar())
            la = label[n, 1:l]
            pred = predictions[n, :]
            metric.update(la, pred)
            metruc_acc.update(la, predictions[n, :(l - 1)])
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
                             transforms=transform_fn
                             )
    val_dataset = CaptionDataSet(image_root="/data3/zyx/yks/coco2017/val2017",
                                 annotation_path="/data3/zyx/yks/coco2017/annotations/captions_val2017.json",
                                 words2index=dataset.words2index,
                                 index2words=dataset.index2words,
                                 transforms=transform_fn
                                 )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            last_batch="discard")
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

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

    net = EncoderDecoder(num_words=num_words, test_max_len=val_dataset.max_len)
    if resume is not None:
        net.collect_params().load(resume, allow_missing=True, ignore_extra=True)
        logger.info("Resumed form checkpoint {}.".format(resume))
    params = net.collect_params()
    for key in params.keys():
        if params[key]._data is not None:
            continue
        else:
            if "bias" in key or "mean" in key or "beta" in key:
                params[key].initialize(init=mx.init.Zero())
                logging.info("initialized {} using Zero.".format(key))
            elif "weight" in key:
                params[key].initialize(init=mx.init.Normal())
                logging.info("initialized {} using Normal.".format(key))
            elif "var" in key or "gamma" in key:
                params[key].initialize(init=mx.init.One())
                logging.info("initialized {} using One.".format(key))
            else:
                params[key].initialize(init=mx.init.Normal())
                logging.info("initialized {} using Normal.".format(key))

    net.collect_params().reset_ctx(ctx=ctx_list)
    trainer = mx.gluon.Trainer(net.collect_params(),
                               'adam',
                               {'learning_rate': 4e-4,
                                'clip_gradient': 5,
                                'multi_precision': True
                                },
                               )
    if trainer_resume is not None:
        trainer.load_states(trainer_resume)
        logger.info("Loaded trainer states form checkpoint {}.".format(trainer_resume))
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
    # net.hybridize(static_alloc=True, static_shape=True)
    net_parallel = DataParallelModel(net, ctx_list=ctx_list, sync=True)
    for nepoch in range(start_epoch, epoches):
        if nepoch > 15:
            trainer.set_learning_rate(4e-5)
        logger.info("Current lr: {}".format(trainer.learning_rate))
        accu_top1_metric.reset()
        accu_top3_metric.reset()
        ctc_loss_metric.reset()
        alpha_metric.reset()
        epoch_bleu.reset()
        batch_bleu.reset()
        for nbatch, batch in enumerate(dataloader):
            batch = [mx.gluon.utils.split_and_load(x, ctx_list) for x in batch]
            inputs = [[x[n] for x in batch] for n, _ in enumerate(ctx_list)]
            losses = []
            with ag.record():
                outputs = net_parallel(*inputs)
                for s_batch, s_outputs in zip(inputs, outputs):
                    image, label, label_len = s_batch
                    predictions, alphas = s_outputs
                    ctc_loss = criterion(predictions, label, label_len)
                    loss2 = 1.0 * ((1. - alphas.sum(axis=1)) ** 2).mean()
                    losses.extend([ctc_loss, loss2])
            ag.backward(losses)
            trainer.step(batch_size=batch_size, ignore_stale_grad=True)
            for n, l in enumerate(label_len):
                l = int(l.asscalar())
                la = label[n, 1:l]
                pred = predictions[n, :(l - 1)]
                accu_top3_metric.update(la, pred)
                accu_top1_metric.update(la, pred)
                epoch_bleu.update(la, predictions[n, :])
                batch_bleu.update(la, predictions[n, :])
            ctc_loss_metric.update(None, preds=nd.sum(ctc_loss) / image.shape[0])
            alpha_metric.update(None, preds=loss2)
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

        bleu, acc_top1 = validate(net, gpu_id=gpu_id,
                                  val_loader=val_loader,
                                  train_index2words=dataset.index2words,
                                  val_index2words=val_dataset.index2words)
        save_path = save_prefix + "_weights-%d-bleu-%.4f-%.4f.params" % (nepoch, bleu, acc_top1)
        net.collect_params().save(save_path)
        trainer.save_states(fname=save_path + ".states")
        logger.info("Saved checkpoint to {}.".format(save_path))


if __name__ == '__main__':
    main()
    # demo()
