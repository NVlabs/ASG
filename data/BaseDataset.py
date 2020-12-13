# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import os
import cv2
import torch
import numpy as np
from pdb import set_trace as bp
import torch.utils.data as data
cv2.setNumThreads(0)


class BaseDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(BaseDataset, self).__init__()
        self._split_name = split_name
        if split_name == 'train':
            self._img_path = setting['train_img_root']
            self._gt_path = setting['train_gt_root']
        elif split_name == 'val':
            self._img_path = setting['val_img_root']
            self._gt_path = setting['val_gt_root']
        elif split_name == 'test':
            self._img_path = setting['test_img_root']
            self._gt_path = setting['test_gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._test_source = setting['test_source'] if 'test_source' in setting else setting['eval_source']
        self._down_sampling = setting['down_sampling_train'] if split_name == 'train' else setting['down_sampling_val']
        print("using downsampling:", self._down_sampling)
        self._file_names = self._get_file_names(split_name)
        print("Found %d images"%len(self._file_names))
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        item_name = names[1].split("/")[-1].split(".")[0]
        img, gt = self._fetch_data(img_path, gt_path)
        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name == 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name), n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path, down_sampling=self._down_sampling[0])
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype, down_sampling=self._down_sampling[1])

        return img, gt

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source
        elif split_name == 'test':
            source = self._test_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            img_name, gt_name = self._process_item_names(item)
            file_names.append([img_name, gt_name])

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        # item = item.split('\t')
        item = item.split(' ')
        img_name = item[0]
        gt_name = item[1]

        return img_name, gt_name

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None, down_sampling=1):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        if isinstance(down_sampling, int):
            try:
                H, W = img.shape[:2]
            except:
                print(img.shape, filepath)
                exit(0)
            if len(img.shape) == 3:
                img = cv2.resize(img, (W // down_sampling, H // down_sampling), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (W // down_sampling, H // down_sampling), interpolation=cv2.INTER_NEAREST)
            assert img.shape[0] == H // down_sampling and img.shape[1] == W // down_sampling
        else:
            assert (isinstance(down_sampling, tuple) or isinstance(down_sampling, list)) and len(down_sampling) == 2
            if len(img.shape) == 3:
                img = cv2.resize(img, (down_sampling[1], down_sampling[0]), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (down_sampling[1], down_sampling[0]), interpolation=cv2.INTER_NEAREST)
            assert img.shape[0] == down_sampling[0] and img.shape[1] == down_sampling[1]

        return img

    @classmethod
    def get_class_colors(*args):
        raise NotImplementedError

    @classmethod
    def get_class_names(*args):
        raise NotImplementedError


if __name__ == "__main__":
    data_setting = {'img_root': '',
                    'gt_root': '',
                    'train_source': '',
                    'eval_source': ''}
    bd = BaseDataset(data_setting, 'train', None)
    print(bd.get_class_names())
