from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class DA(object):

    def __init__(self, data_dir, source, target, source_extension, target_extension):

        # source / target image root
        self.source_images_dir = osp.join(data_dir, source)
        self.target_images_dir = osp.join(data_dir, target)
        self.source_extension = source_extension
        self.target_extension = target_extension
        # training image dir
        # self.source_train_path = 'bounding_box_train'
        self.source_train_path = 'train'

        # self.target_train_path = 'bounding_box_train'
        # self.target_train_camstyle_path = 'bounding_box_train_camstyle'
        self.gallery_path = 'test'
        self.query_path = 'query'

        #self.source_train, self.target_train, self.query, self.gallery = [], [], [], []
        self.source_train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        # self.cam_dict = self.set_cam_dict()
        # self.target_num_cam = self.cam_dict[target]
        # self.source_num_cam = self.cam_dict[source]

        self.load()

    def set_cam_dict(self):
        cam_dict = {}
        cam_dict['market'] = 6
        cam_dict['duke'] = 8
        cam_dict['msmt17'] = 15
        cam_dict['market2s01'] = 6
        cam_dict['s012market'] = 6
        cam_dict['sys'] = 2
        return cam_dict

    def preprocess(self, images_dir, path, img_extension='jpg', relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        ret = []
        img_extension = '*.' + img_extension
        fpaths = sorted(glob(osp.join(images_dir,path,img_extension)))
        # if 'cuhk03' in images_dir or 's012market' in images_dir:
        #     fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        # else:
        #     fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if 'cuhk03' in images_dir:
                name = osp.splitext(fname)[0]
                pid, cam = map(int, pattern.search(fname).groups())
                # bag, pid, cam, _ = map(int, name.split('_'))
                # pid += bag * 1000
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))
    def sys_preprocess(self, images_dir, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        # if ('cuhk03' in images_dir) or ('sys' in images_dir):
        #     fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        # else:
        #     fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            # if ('cuhk03' in images_dir):
            #     name = osp.splitext(fname)[0]
            #     pid, cam = map(int, pattern.search(fname).groups())
            #     # bag, pid, cam, _ = map(int, name.split('_'))
            #     # pid += bag * 1000
            # elif 'sys' in images_dir:
            #     print(fname)
            #     pid = int(fname.split('_')[0])
            #     cam = int(fname.split('_')[1])
            #     #sys_pattern = re.compile(r'([-\d]+)_([-\d]+)')
            #     #pid, cam = map(int, sys_pattern.search(fname).group())
            # else:
            #     pid, cam = map(int, pattern.search(fname).groups())
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.source_train, self.num_train_ids = self.preprocess(self.source_images_dir, self.source_train_path,img_extension=self.source_extension)
        # self.source_train, self.num_train_ids = self.preprocess(self.source_images_dir, self.source_train_path)
        # self.target_train, _ = self.preprocess(self.target_images_dir, self.target_train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.target_images_dir, self.gallery_path,img_extension=self.target_extension, relabel=False)
        # self.gallery, self.num_gallery_ids = self.sys_preprocess(self.target_images_dir, self.gallery_path, False)
        # self.query, self.num_query_ids = self.sys_preprocess(self.target_images_dir, self.query_path, False)
        self.query, self.num_query_ids = self.preprocess(self.target_images_dir, self.query_path, img_extension=self.target_extension, relabel=False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  source train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.source_train)))
        # print("  target train    | 'Unknown' | {:8d}"
        #       .format(len(self.target_train)))


        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
        print('gallery path : %s' % str(osp.join(self.target_images_dir,self.gallery_path)))
        print('query path : %s' % str(osp.join(self.target_images_dir,self.query_path)))
