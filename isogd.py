import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import random
import numpy as np
import h5py
import math

import cv2
from utils import mask_unset, get_hand_rects

class ScaleRect(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h
    def __call__(self, img):
        return img.resize((self.w, self.h))

class RandomScaleOffsetCropFlip(object):
    def __init__(self, W=320, H=256, scale_rates=(1.0, 0.875,), max_distort=1.0, crop_w=224, crop_h=224,
                 fix_offset=True, more_offset=False):
        self.W = W
        self.H = H
        self.scale_rates = scale_rates
        self.max_distort = max_distort
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.fix_crop = fix_offset
        self.more_offset = more_offset

        self._get_scale_list()
        self.get_offset_list()

    def _get_scale_list(self):
        W, H = self.W, self.H

        min_size = min(W, H)

        scale_crop_size_list = []
        for w in self.scale_rates:
            scale_crop_w = int(min_size * w)
            for h in self.scale_rates:
                if abs(h-w) > self.max_distort:
                    continue

                scale_crop_h = int(min_size * h)
                scale_crop_size_list.append((scale_crop_w, scale_crop_h))
        self.scale_crop_sizes = scale_crop_size_list

    def get_offset_list(self):
        if self.more_offset:
            self.offsetrs = [(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0), (2.0, 2.0),
                             (0.0, 2.0), (4.0, 2.0), (2.0, 0.0), (2.0, 4.0),
                             (1.0, 1.0), (1.0, 3.0), (3.0, 1.0), (3.0, 3.0)]
        else:
            self.offsetrs = [(2.0, 2.0),
                             (1.0, 1.0), (1.0, 3.0), (3.0, 1.0), (3.0, 3.0)]

    def set_rnd(self):
        self.rand_scale_w, self.rand_scale_h = random.choice(self.scale_crop_sizes)
        self.rand_offset_wr, self.rand_offset_hr = random.choice(self.offsetrs)
        self.rand_flip = random.randint(0, 1)

    def __call__(self, img):
        img = img.resize((self.W, self.H))
        rand_scale_w, rand_scale_h = self.rand_scale_w, self.rand_scale_h
        rand_offset_wr, rand_offset_hr = self.rand_offset_wr, self.rand_offset_hr

        dw = (self.W - rand_scale_w)/4.0
        dh = (self.H - rand_scale_h)/4.0

        rand_offset_w = int(dw * rand_offset_wr)
        rand_offset_h = int(dh * rand_offset_hr)
        img = img.crop((rand_offset_w, rand_offset_h, rand_offset_w+rand_scale_w, rand_offset_h+rand_scale_h)).resize((self.crop_w, self.crop_h))
        if self.rand_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

def default_get_frame_inds(n_frame, n_seg, seq_strides, seq_len):
    assert n_frame > 0
    rand_stride = random.choice(seq_strides)
    # use linspace to set boundaries of segs
    end_points = np.linspace(0, n_frame - 1, n_seg + 1).round().astype(np.int).tolist()

    frame_inds = []
    for seg_ind in range(n_seg):
        if rand_stride == 'center_random':
            assert n_seg == 3
            if seg_ind != 1:
                continue
        seg_start = end_points[seg_ind]
        seg_end = end_points[seg_ind+1]
        if isinstance(rand_stride, int):
            strided_seg = range(seg_start, seg_end + 1, rand_stride)  # at least one sample here
        elif rand_stride == 'random' or rand_stride == 'center_random':
            strided_seg = sorted(np.unique(np.random.randint(seg_start, seg_end+1, seq_len)).tolist())
        elif rand_stride == 'uniform':
            strided_seg = np.arange(seg_start, seg_end+1)
            seg_len = len(strided_seg)
            strided_seg = strided_seg[np.linspace(0, seg_len - 1, seq_len).round().astype(np.int).tolist()].tolist()
        elif rand_stride == 'center':
            strided_seg = np.arange(seg_start, seg_end+1)
            if len(strided_seg) > seq_len:
                offset = (len(strided_seg) - seq_len) / 2
                strided_seg = strided_seg[offset:offset+seq_len]
            strided_seg = strided_seg.tolist()

        else:
            raise ValueError('stride mode wrong')

        rand_start = random.randrange(0, max(len(strided_seg) - seq_len + 1, 1))
        sample_inds = strided_seg[rand_start:rand_start+seq_len]
        if len(sample_inds) < seq_len:
            sample_inds = np.pad(sample_inds, (0, seq_len - len(sample_inds)), mode='wrap')
            sample_inds = sample_inds.tolist()
        frame_inds += sample_inds
    frame_inds = np.asarray(frame_inds)
    return frame_inds

class ISOGD(Dataset):
    def __init__(self, info_basedir='', mode='train',
                 video_basedir='',
                 to_read=('label', 'rgb_h5'),
                 seq_len=32, n_seg=1, seq_strides=('random', ),
                 aug_video=True, transformer=None,
                 run_n_sample=0, shuffle=True,
                 vis_pose=False):
        super(ISOGD, self).__init__()
        self.info_basedir = info_basedir
        self.mode = mode
        self.seq_len = seq_len
        self.n_seg = n_seg
        self.seq_strides = seq_strides
        self.aug_video = aug_video

        self.to_read = to_read
        self.transformer = transformer
        self.shuffle = shuffle
        self.vis_pose = vis_pose

        phase_str = '1' if mode in ['train', 'val'] else '2'

        self.rgb_avi_basedir = '{}/IsoGD_phase_{}/'.format(video_basedir, phase_str)
        self.depth_avi_basedir = '{}/IsoGD_phase_{}/'.format(video_basedir, phase_str)
        self.pose_h5_basedir = './pose_h5/phase{}'.format(phase_str)

        with open(os.path.join(info_basedir, '{}_{}_info.json'.format(mode, 'rgb')), 'r') as f:
            self.rgb_info = json.load(f)

        with open(os.path.join(info_basedir, '{}_{}_info.json'.format(mode, 'd')), 'r') as f:
            self.depth_info = json.load(f)

        if run_n_sample == 0:
            run_n_sample = len(self.rgb_info)

        self.run_n_sample = run_n_sample
        self.n_sample = len(self.rgb_info)

        n_epoch = int(math.ceil(self.run_n_sample * 1.0 / self.n_sample))

        n_sample_ind_iter = []
        for _ in range(n_epoch):
            if self.shuffle:
                iter_epoch = np.random.permutation(self.n_sample).tolist()
            else:
                iter_epoch = range(self.n_sample)
            n_sample_ind_iter = n_sample_ind_iter + iter_epoch
        n_sample_ind_iter = n_sample_ind_iter[:self.run_n_sample]
        self.n_sample_ind_iter = n_sample_ind_iter

    def get_rgb_random_frame_inds(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.rgb_info[item]
        n_frame = info['n_frame']

        frame_inds = default_get_frame_inds(n_frame=n_frame, n_seg=self.n_seg, seq_strides=self.seq_strides,
                                            seq_len=self.seq_len)

        context['rgb_frame_inds'] = frame_inds
        return frame_inds

    def get_depth_random_frame_inds(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.depth_info[item]
        n_frame = info['n_frame']

        frame_inds = default_get_frame_inds(n_frame=n_frame, n_seg=self.n_seg, seq_strides=self.seq_strides,
                                            seq_len=self.seq_len)

        context['depth_frame_inds'] = frame_inds
        return frame_inds

    def get_label(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.depth_info[item]
        if 'label' in info:
            label = info['label']
        else:
            label = 0
        return label

    def get_rgb_vid_name(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.rgb_info[item]
        vid_name = info['vid_name']
        return vid_name

    def get_depth_vid_name(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.depth_info[item]
        vid_name = info['vid_name']
        return vid_name

    def _get_rgb_avi_np(self, item, context):
        if 'rgb_frame_inds' not in context:
            context['rgb_frame_inds'] = self.get_rgb_random_frame_inds(item, context)

        item = self.n_sample_ind_iter[item]
        info = self.rgb_info[item]
        vid_name = info['vid_name']

        rgb_avi_path = os.path.join(self.rgb_avi_basedir, vid_name + '.avi')
        frame_inds = context['rgb_frame_inds']

        vid = cv2.VideoCapture(rgb_avi_path)

        ind = 0
        i = 0
        rgb_np = None
        while True:
            flag = vid.grab()
            if not flag:
                break
            if ind in frame_inds:
                flag, img = vid.retrieve()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if rgb_np is None:
                    rgb_np = np.zeros(([len(frame_inds), img.shape[0], img.shape[1], 3]), np.uint8)
                rgb_np[i] = img
                i += 1
            ind += 1

        if len(rgb_np) != len(frame_inds):
            raise RuntimeError('frame length wrong!')

        context['rgb_np'] = rgb_np
        return rgb_np

    def _get_depth_avi_np(self, item, context):
        if 'depth_frame_inds' not in context:
            context['depth_frame_inds'] = self.get_depth_random_frame_inds(item, context)

        item = self.n_sample_ind_iter[item]
        info = self.depth_info[item]
        vid_name = info['vid_name']

        depth_avi_path = os.path.join(self.depth_avi_basedir, vid_name + '.avi')
        frame_inds = context['depth_frame_inds']

        vid = cv2.VideoCapture(depth_avi_path)
        ind = 0
        i = 0
        depth_np = None
        while True:
            flag = vid.grab()
            if not flag:
                break
            if ind in frame_inds:
                flag, img = vid.retrieve()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if depth_np is None:
                    depth_np = np.zeros(([len(frame_inds), img.shape[0], img.shape[1], 3]), np.uint8)
                depth_np[i] = img
                i += 1
            ind += 1
        if len(depth_np) != len(frame_inds):
            raise RuntimeError('frame length wrong!')

        context['depth_np'] = depth_np
        return depth_np

    def get_rgb_avi(self, item, context):
        if 'rgb_np' not in context:
            self._get_rgb_avi_np(item, context)

        frame_inds = context['rgb_frame_inds']

        img_seq = None

        if 'img' in self.transformer:
            for t in self.transformer['img'].transforms:
                if hasattr(t, 'set_rnd'):
                    t.set_rnd()

        for i, frame_ind in enumerate(frame_inds):
            img = context['rgb_np'][i]

            img = Image.frombuffer('RGB', img.shape[1::-1], img, 'raw', 'RGB', 0, 1)

            if 'img' in self.transformer:
                if not self.aug_video:
                    for t in self.transformer['img'].transforms:
                        if hasattr(t, 'set_rnd'):
                            t.set_rnd()
                img = self.transformer['img'](img)  # already of size 3 x H x W

            if img_seq is None:
                shape = list(img.size())
                img_seq = torch.zeros([len(frame_inds)] + shape)

            img_seq[i] = img

        return img_seq

    def _mask_body(self, imgs, poses):

        lboxes, rboxes = get_hand_rects(imgs, poses)
        mask_imgs = []
        for lbox, rbox, img in zip(lboxes, rboxes, imgs):
            img = mask_unset([lbox, rbox], img)
            mask_imgs.append(img)
        return mask_imgs

    def get_focus_rgb(self, item, context):
        if 'rgb_np' not in context:
            self._get_rgb_avi_np(item, context)
        if 'pose' not in context:
            self.get_pose(item, context)

        img_seq = None
        frame_inds = context['rgb_frame_inds']
        poses = context['pose']
        imgs = context['rgb_np']

        poses = poses[:, :, :2]
        imgs = self._mask_body(imgs, poses)

        if 'img' in self.transformer:
            for t in self.transformer['img'].transforms:
                if hasattr(t, 'set_rnd'):
                    t.set_rnd()

        for i, frame_ind in enumerate(frame_inds):
            try:
                img = imgs[i]
            except:
                print(len(frame_inds), len(imgs), self.rgb_info[self.n_sample_ind_iter[item]])

            img = Image.frombuffer('RGB', img.shape[1::-1], img, 'raw', 'RGB', 0, 1)

            if 'img' in self.transformer:
                if not self.aug_video:
                    for t in self.transformer['img'].transforms:
                        if hasattr(t, 'set_rnd'):
                            t.set_rnd()
                img = self.transformer['img'](img)  # already of size 3 x H x W

            if img_seq is None:
                shape = list(img.size())
                img_seq = torch.zeros([len(frame_inds)] + shape)

            img_seq[i] = img

        return img_seq

    def get_depth_avi(self, item, context):
        if 'depth_np' not in context:
            self._get_depth_avi_np(item, context)

        frame_inds = context['depth_frame_inds']

        img_seq = None

        if 'img' in self.transformer:
            for t in self.transformer['img'].transforms:
                if hasattr(t, 'set_rnd'):
                    t.set_rnd()

        for i, frame_ind in enumerate(frame_inds):
            img = context['depth_np'][i]

            img = Image.frombuffer('RGB', img.shape[1::-1], img, 'raw', 'RGB', 0, 1)

            if 'img' in self.transformer:
                if not self.aug_video:
                    for t in self.transformer['img'].transforms:
                        if hasattr(t, 'set_rnd'):
                            t.set_rnd()
                img = self.transformer['img'](img)  # already of size 3 x H x W

            if img_seq is None:
                shape = list(img.size())
                img_seq = torch.zeros([len(frame_inds)] + shape)

            img_seq[i] = img

        return img_seq

    def get_focus_depth(self, item, context):
        if 'depth_np' not in context:
            self._get_depth_avi_np(item, context)
        if 'pose' not in context:
            self.get_pose(item, context)

        img_seq = None
        frame_inds = context['depth_frame_inds']
        poses = context['pose']
        imgs = context['depth_np']

        poses = poses[:, :, :2]
        imgs = self._mask_body(imgs, poses)

        if 'img' in self.transformer:
            for t in self.transformer['img'].transforms:
                if hasattr(t, 'set_rnd'):
                    t.set_rnd()

        for i, frame_ind in enumerate(frame_inds):
            try:
                img = imgs[i]
            except:
                print(len(frame_inds), len(imgs), self.rgb_info[self.n_sample_ind_iter[item]])

            img = Image.frombuffer('RGB', img.shape[1::-1], img, 'raw', 'RGB', 0, 1)

            if 'img' in self.transformer:
                if not self.aug_video:
                    for t in self.transformer['img'].transforms:
                        if hasattr(t, 'set_rnd'):
                            t.set_rnd()
                img = self.transformer['img'](img)  # already of size 3 x H x W

            if img_seq is None:
                shape = list(img.size())
                img_seq = torch.zeros([len(frame_inds)] + shape)

            img_seq[i] = img

        return img_seq

    def get_pose(self, item, context):
        item = self.n_sample_ind_iter[item]
        info = self.rgb_info[item]
        vid_name = info['vid_name']

        pose_h5_path = os.path.join(self.pose_h5_basedir, vid_name + '.h5')
        frame_inds = context['rgb_frame_inds']

        with h5py.File(pose_h5_path, 'r') as f:
            pose_list = f['pose'][()]
        # T x 18 x 2
        pose_list = pose_list[frame_inds]
        context['pose'] = pose_list
        return pose_list

    def __getitem__(self, item):
        context = dict()
        rgb_frame_inds = self.get_rgb_random_frame_inds(item, context=context)
        depth_frame_inds = self.get_depth_random_frame_inds(item,  context=context)

        output = []
        for key in self.to_read:
            method = getattr(self, 'get_{}'.format(key))
            val = method(item, context)
            output.append(val)
        return output

    def __len__(self):
        return self.run_n_sample


