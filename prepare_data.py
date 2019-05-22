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
import tqdm
from resnetv1b import resnet50_v1b
import h5py

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


class DirectoryDataSet():
    def __init__(self, image_root, transforms=None):
        images = os.listdir(image_root)
        self._transforms = transforms
        self.images = images
        self.image_root = image_root

    def __getitem__(self, item):
        return item, self._transforms(cv2.imread(os.path.join(self.image_root, self.images[item]))[:, :, ::-1])

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    gpu_id = 8
    net = Encoder()
    net.collect_params().reset_ctx(mx.gpu(gpu_id))
    from mxnet.gluon.data.vision import transforms

    transform_fn = transforms.Compose([
        LeftTopPad(dest_shape=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset = DirectoryDataSet(image_root="/data3/zyx/yks/coco2017/train2017", transforms=transform_fn)
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    f = h5py.File('output/train2017.h5', 'w')
    for batch in tqdm.tqdm(loader):
        indices, data = batch
        outputs = net(data.as_in_context(mx.gpu(gpu_id))).asnumpy()
        indices = indices.asnumpy()
        for idx, output in zip(indices, outputs):
            file_name = dataset.images[idx]
            f[file_name] = output
    f.close()
