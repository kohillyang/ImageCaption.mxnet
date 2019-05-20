import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from itertools import chain
import mxnet as mx


class LeftTopPad(mx.image.Augmenter):
    def __init__(self, dest_shape=(768, 768)):
        super(LeftTopPad, self).__init__()
        self.dest_shape = dest_shape

    def __call__(self, image, *args):
        if isinstance(image, mx.nd.NDArray):
            image = image.asnumpy()
        dshape = self.dest_shape
        fscale = min(dshape[0] / image.shape[0], dshape[1] / image.shape[1])

        img_resized = cv2.resize(image, dsize=(0, 0), fx=fscale, fy=fscale)  # type: np.ndarray
        img_padded = np.zeros(shape=(int(dshape[0]), int(dshape[1]), 3), dtype=np.float32)
        h, w, c = img_resized.shape
        # h_p, w_p, c_p = img_padded.shape
        start_h = 0
        start_w = 0
        img_padded[start_h:(start_h + h), start_w:(start_w + w), :] = img_resized

        return mx.nd.array(img_padded)


def convert_dataset(annotation_path):
    obj = json.load(open(annotation_path, "rb"))
    image_id2file_name = {}
    for image in obj["images"]:
        image_id2file_name[image["id"]] = image["file_name"]

    all_captions = [x["caption"] for x in obj["annotations"]]
    all_captions = [x.replace(".", "").strip().split() for x in all_captions]
    print("Creating dictionaries, this can take some time...")
    all_words = set()
    for sen in all_captions:
        all_words.update(sen)
    print("Creating dictionaries finished.")

    words2index = {}
    index2words = {}
    for c, w in enumerate(chain(["<START>", "<END>"], all_words, ["<PAD>"])):
        word_index = c 
        words2index[w] = word_index
        index2words[word_index] = w

    objs = {}
    objs["images"] = []

    for one_caption in tqdm(obj["annotations"]):
        oneimg = {
            "sentences": [{"tokens": one_caption["caption"].replace(".", "").strip().split(),
                           "imgid": one_caption["image_id"]}],
            "filename": image_id2file_name[one_caption["image_id"]]
        }
        objs["images"].append(oneimg)
    return objs, words2index, index2words


class CaptionDataSet(mx.gluon.data.Dataset):
    def __init__(self, image_root, annotation_path, index2words=None, words2index=None, transforms=None):
        super(CaptionDataSet, self).__init__()
        objs, wi, iw = convert_dataset(annotation_path)
        self.objs = list(filter(lambda x: len(x["sentences"][0]["tokens"]) >= 1, objs["images"]))
        if index2words is not None and words2index is not None:
            self.index2words = index2words
            self.words2index = words2index
        else:
            self.index2words = iw
            self.words2index = wi
        self._max_len = max([len(x["sentences"][0]["tokens"]) for x in self.objs]) + 2
        self.transforms = transforms
        self._image_root = image_root

    @property
    def max_len(self):
        return self._max_len

    @property
    def words_count(self):
        return int(self.words2index["<PAD>"]) + 1  # start from 0.

    def __getitem__(self, idx):
        filepath, image, sentences = self.at_with_image_path(idx)
        label = [self.words2index[x] if x in self.words2index else self.words2index["<PAD>"] for x in sentences]
        label.insert(0, self.words2index["<START>"])
        label.append(self.words2index["<END>"])
        label_len = len(label)
        assert len(label) <= self.max_len
        label = np.array(label)
        label_padded = np.empty((self.max_len,), dtype=np.int32)
        label_padded.fill(0)
        label_padded[:label.shape[0]] = label
        if self.transforms is not None:
            image = mx.nd.array(image)
            image = self.transforms(image)
        assert label_len > 1
        return image.asnumpy(), label_padded.astype(np.float32), np.array([label_len]).astype(np.float32)

    def at_with_image_path(self, idx):
        oneimg = self.objs[idx]
        filename = oneimg["filename"]
        filepath = os.path.join(self._image_root, filename)
        sentences = oneimg["sentences"][0]["tokens"]
        return filepath, cv2.imread(filepath)[:, :, ::-1], sentences

    def __len__(self):
        return len(self.objs)
