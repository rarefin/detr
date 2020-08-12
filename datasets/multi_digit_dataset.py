import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import torch
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class MultiDigitDataset(data.Dataset):
    def __init__(self, data_dir, split='train', transform=None):

        self.transform = transform
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split, "normal")
        self.img_dir = self.split_dir + "/imgs/"
        self.max_objects = 3

        self.filenames = self.load_filenames()
        self.bboxes = self.load_bboxes()
        self.labels = self.load_labels()

    def get_img(self, img_path):
        img = Image.open(img_path)

        return img

    def load_bboxes(self):
        bbox_path = os.path.join(self.split_dir, 'bboxes.pickle')
        with open(bbox_path, "rb") as f:
            bboxes = pickle.load(f, encoding='latin1')
            bboxes = np.array(bboxes, dtype=np.float)
        return bboxes

    def load_labels(self):
        label_path = os.path.join(self.split_dir, 'labels.pickle')
        with open(label_path, "rb") as f:
            labels = pickle.load(f, encoding='latin1')
            labels = np.argmax(labels, axis=-1)
            labels = np.array(labels)
        return labels

    def load_filenames(self):
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f, encoding='latin1')
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        # load image
        key = self.filenames[index]
        key = key.split("/")[-1]
        img_name = self.split_dir + "/imgs/" + key
        img = self.get_img(img_name)

        # load bbox
        bbox = self.bboxes[index].astype(np.float)

        # guard against no boxes via resizing
        w, h = img.size
        boxes = torch.as_tensor(bbox, dtype=torch.float32).reshape(-1, 4) * torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # load label
        labels = self.labels[index]

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels, dtype=torch.int64)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

        # return img, target["boxes"], target["labels"]

    def __len__(self):
        return len(self.filenames)
