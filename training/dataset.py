# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2

import pyspng


# try:
#     import pyspng
# except ImportError:
#     pyspng = None


# ----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            name,  # Name of the dataset.
            raw_shape,  # Shape of the raw image data (NCHW).
            max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
            use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
            xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
            random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels

        self._raw_labels = None
        self._label_shape = None
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # We don't Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._w[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
class ImageFolderBaseDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        load_normal_map, # read normal map or not
        white_bg,
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.load_normal_map = load_normal_map
        self.white_bg = white_bg

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION and not any(substr in fname for substr in ['albedo', 'normal', 'id']))
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        self.img_size = resolution
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.png':
            if self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
            # image = image[..., :3] * (image[..., -1:] == 255) + (255. - image[..., -1:])
            assert image.shape[-1]==4
            image = image[..., :3] * (image[..., -1:] == 255) + (255. - image[..., -1:])
            image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
            image = image.resize((self.img_size, self.img_size))
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'extrinsics_smpl.json'

        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
#----------------------------------------------------------------------------
class CameraSMPLDataset(ImageFolderBaseDataset):
    def __getitem__(self, idx):
        #image, mask = self._load_image_and_mask(self._raw_idx[idx])
        if self.load_normal_map:    
            image, mask, normal_image = self._load_rgb_and_normal_and_mask(self._raw_idx[idx])
        else:
            image, mask = self._load_image_and_mask(self._raw_idx[idx])
            
        label = self.get_label(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape

        """
        label info: 
            0:16    world2camera_matrix shape (16,)
            16:19   global_orient       shape (3,)
            19:88   body_pose           shape (69,)
            88:98   betas               shape (10,)
        """

        assert label.shape == (98,)

        if self.load_normal_map:    
            return image.copy(), label, mask.copy(), normal_image.copy()
        else:
            return image.copy(), label, mask.copy(), []

    def _load_image_and_mask(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        with self._open_file(fname) as f:
            if self._file_ext(fname) == '.png':
                ori_img = pyspng.load(f.read())
            else:
                ori_img = np.array(PIL.Image.open(f))

        assert ori_img.shape[-1] == 4
        img = ori_img[:, :, :3]
        mask = ori_img[:, :, 3:4]

        image = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
        resize_img = np.array(image.resize((self.img_size, self.img_size)))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)  ########

        img = resize_img.transpose(2, 0, 1)
        #background = np.zeros_like(img)
        if self.white_bg:
            background = np.ones_like(img) * 255
        else:
            background = np.zeros_like(img)

        img = img * (mask > 0).astype(np.float) + background * (1 - (mask > 0).astype(np.float))
        return np.ascontiguousarray(img),  np.ascontiguousarray(mask)

    # def _load_image_and_mask(self, raw_idx):
    def _load_rgb_and_normal_and_mask(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._file_ext(fname) == '.png':
                ori_img = pyspng.load(f.read())
            else:
                ori_img = np.array(PIL.Image.open(f))
        
        normal_fname = fname.replace('.png', '_normal.png')
        with self._open_file(normal_fname) as f:
            if self._file_ext(normal_fname) == '.png':
                ori_normal_img = pyspng.load(f.read())
            else:
                ori_normal_img = np.array(PIL.Image.open(f))

        assert ori_img.shape[-1] == 4
        img = ori_img[:, :, :3]
        mask = ori_img[:, :, 3:4]

        normal_img = ori_normal_img[:, :, :3]

        image = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
        resize_img = np.array(image.resize((self.img_size, self.img_size)))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)  ########

        normal_image = PIL.Image.fromarray(normal_img.astype('uint8'), 'RGB')
        resize_normal_img = np.array(normal_image.resize((self.img_size, self.img_size)))

        img = resize_img.transpose(2, 0, 1)
        #background = np.zeros_like(img)
        if self.white_bg:
            background = np.ones_like(img) * 255
        else:
            background = np.zeros_like(img)
        img = img * (mask > 0).astype(np.float) + background * (1 - (mask > 0).astype(np.float))

        normal_img = resize_normal_img.transpose(2, 0, 1)
        normal_background = np.ones_like(img) * 127.5
        normal_img = normal_img * (mask > 0).astype(np.float) + normal_background * (1 - (mask > 0).astype(np.float))
        
        return np.ascontiguousarray(img),  np.ascontiguousarray(mask), np.ascontiguousarray(normal_img)

    def _load_raw_labels(self):
        fname = 'extrinsics_smpl.json'

        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)

        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
#----------------------------------------------------------------------------


