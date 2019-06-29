from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import pdb

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap, map_cmc
from .utils.meters import AverageMeter
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import os.path as osp
from PIL import Image
from torchvision.transforms import functional as F
import pdb
import visdom


def extract_cnn_feature(model, inputs, opt, output_feature=None):
    model.eval()
    # device = torch.device('cuda:'+str(opt.gpuid))
    inputs = to_torch(inputs)
    inputs = inputs.cuda()
    outputs = model(inputs, output_feature)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=1, output_feature=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, output_feature)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    # We use clamp to keep numerical stability
    dist = torch.clamp(dist, 1e-8, np.inf)
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Evaluation
    mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, all_cmc[k - 1]))
    return

def visualization(dist, query, gallery,query_path, gallery_path, output_dir='/home/ltb/myshare/ECN/data/'):
    dist = dist.cpu().numpy()
    indices = np.argsort(dist, axis=1)
    for q_index in range(50):
        index = indices[q_index]
        matplotlib.use('agg')
        fig = plt.figure(figsize=(30, 10))
        ax = plt.subplot(2, 6, 1)
        ax.axis('off')
        path = osp.join(query_path, query[q_index][0])
        print(path)
        ax.set_title(query[q_index][0])

        plt.imshow(plt.imread(path))
        for i in range(10):
            if i < 5:
                ax = plt.subplot(2, 6, i + 2)
            else:
                ax = plt.subplot(2, 6, i + 3)
            ax.axis('off')
            path = osp.join(gallery_path, gallery[index[i]][0])
            ax.set_title(gallery[index[i]][0])
            plt.imshow(plt.imread(path))
            print(path)
        fig.savefig(output_dir + str(q_index) + 'g2g')
    # Traditional evaluation
    # Compute mean AP
    # mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))
    #
    # # Compute CMC scores
    # cmc_configs = {
    #     'market1501': dict(separate_camera_set=False,
    #                        single_gallery_shot=False,
    #                        first_match_break=True)}
    # cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
    #                         query_cams, gallery_cams, **params)
    #               for name, params in cmc_configs.items()}
    #
    # print('CMC Scores')
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}'
    #           .format(k, cmc_scores['market1501'][k - 1]))
    #
    # return cmc_scores['market1501'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
    def evaluate(self, query_loader, gallery_loader, query, gallery,  print_freq=1,output_feature=None):
        query_features, _ = extract_features(self.model, query_loader, print_freq, output_feature)
        gallery_features, _ = extract_features(self.model, gallery_loader, print_freq, output_feature)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        # visualization(distmat,query,gallery,query_path='/ssd4/ltb/datasets/reid/sys01_market/market/query',gallery_path='/ssd4/ltb/datasets/reid/sys01_market/market/test')
        return evaluate_all(distmat, query=query, gallery=gallery)

