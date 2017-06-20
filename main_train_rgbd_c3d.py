import torch.multiprocessing
import argparse
import time
import numpy as np
import torch.nn.utils
import torch.nn.functional as F
import datetime
import os
parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--resume', default='', type=str)

parser.add_argument('--seq_len', default=32, type=int)
parser.add_argument('--modality', default='focus_rgb', type=str)

parser.add_argument('--lr_decay_step', default=-1, type=int)
parser.add_argument('--max_step', default=-1, type=int)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--save_every_step', default=400, type=float)

parser.add_argument('--evaluate', default=0, type=int)

parser.add_argument('--eval_split', default='val', type=str)
parser.add_argument('--info_basedir', default='./dataset/info', type=str)
parser.add_argument('--video_basedir', default='./dataset/', type=str)


args = parser.parse_args()

if args.max_step == -1:
    args.max_step = 4100 * 4

if args.lr_decay_step == -1:
    args.lr_decay_step = 8000

args.save_every_step = args.max_step/25

args.gpu_id = args.gpu_id.split(',')
args.gpu_id = [int(r) for r in args.gpu_id]

import torch
torch.cuda.set_device(args.gpu_id[0])

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd.variable import Variable


from isogd import ISOGD, RandomScaleOffsetCropFlip, ScaleRect
from rgbd_c3d import RGBDC3D

step = 0

def main():
    global step, model, crit, optimizer
    global train_dataset, test_dataset, train_loader, test_loader
    # prepare model
    print('creating model...')
    model, crit, optimizer = prepare_model()
    # optionally resume from a checkpoint
    if args.resume != '':
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    print('prepare dataset...')
    (train_dataset, train_loader), (test_dataset, test_loader) = prepare_dataset()

    if args.evaluate:
        print('test...')
        test_dataset.seq_strides = ('center', )
        validate(is_test=True)
        return
    print('start training...')

    train()

def prepare_dataset():
    train_transformer = transforms.Compose([
                                   RandomScaleOffsetCropFlip(W=171, H=128, scale_rates=(1.0, 0.875, 0.75), crop_w=112, crop_h=112),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[1., 1., 1.]),
                               ])
    test_transformer = transforms.Compose([
                               ScaleRect(171, 128),
                               transforms.CenterCrop(112),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[1., 1., 1.]),
                               ])

    if args.modality == 'focus_rgb':
        to_read = ('label', 'focus_rgb')
    elif args.modality == 'focus_depth':
        to_read = ('label', 'focus_depth')
    else:
        raise RuntimeError('wrong modality')

    seq_strides = (1,)

    info_basedir = args.info_basedir
    test_mode = args.eval_split
    train_dataset = ISOGD(info_basedir=info_basedir, mode='train', to_read=to_read,
                          seq_len=args.seq_len, n_seg=1, seq_strides=seq_strides,
                          aug_video=True, transformer=dict(img=train_transformer),
                          run_n_sample=args.max_step*args.batch_size, shuffle=True,
                          video_basedir=args.video_basedir)

    if args.modality == 'focus_rgb':
        to_read = ('rgb_vid_name', 'label', 'focus_rgb')
    elif args.modality == 'focus_depth':
        to_read = ('depth_vid_name', 'label', 'focus_depth')
    else:
        raise RuntimeError('wrong modality')

    test_dataset = ISOGD(info_basedir=info_basedir, mode=test_mode, to_read=to_read,
                         seq_len=args.seq_len, n_seg=1, seq_strides=('center', ),
                         aug_video=True, transformer=dict(img=test_transformer),
                         run_n_sample=0, shuffle=False,
                         video_basedir=args.video_basedir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                              pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                              pin_memory=False)

    return (train_dataset, train_loader), (test_dataset, test_loader)


def prepare_model():
    model = RGBDC3D(cnn_name='resnet18_c3d', feature_name='layer4', cnn_dropout=0.5, modality=args.modality,
                    seq_len=args.seq_len,
                    gpu_id=args.gpu_id)
    crit = torch.nn.CrossEntropyLoss()

    model = model.cuda(args.gpu_id[0])

    optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    return model, crit, optimizer


def train():
    global step, best_prec1
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    #
    end = time.time()
    for iter_, (labels, imgs) in enumerate(train_loader):

        model.train()
        data_time.update(time.time()-end)
        target_var = Variable(labels.cuda(args.gpu_id[0]))

        # forward
        ys, = model([imgs])

        loss = crit(ys, target_var)

        loss_meter.update(loss.data[0], len(imgs))

        prec1, prec5 = accuracy(ys.data.cpu(), labels, topk=(1, 5))

        top1_meter.update(prec1[0], len(imgs))
        top5_meter.update(prec5[0], len(imgs))

        # back
        torch.autograd.backward([loss], [loss.data.new(1).fill_(1.0)])

        # compute gradient and do SGD step
        lr = step_adjust_learning_rate(optimizer=optimizer, lr0=args.lr, step=step, step_size=args.lr_decay_step, gamma=0.1)


        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)

        epoch = step / max(train_loader.dataset.n_sample/args.batch_size, 1)
        if step % 1 == 0:
            print('time: {time} \t '
                'Epoch: [{0}][{1}/{2}/{step}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'lr {lr:.6f}\t'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=loss_meter, top1=top1_meter, top5=top5_meter, lr=lr, step=step,
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        # validate
        if (step % args.save_every_step == 0 or step >= args.max_step-1) and step != 0:
            prec1 = validate()
            # remember best prec@1 and save checkpoint
            if args.resume == '':
                model_name = '{}.model'.format(args.modality)
            elif args.modality == 'focus_rgb':
                model_name = 'focus_rgb_init_depth.model'
            elif args.modality == 'focus_depth':
                model_name = 'focus_depth_init_rgb.model'
            else:
                raise RuntimeError('modality wrong!')

            torch.save({'model_state_dict': model.state_dict()}, './model/{}'.format(model_name))
            if step >= args.max_step:
                break

        if step >= args.max_step:
            break
        step += 1
        end = time.time()
    return step

def validate(is_test=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    #
    model.eval()

    end = time.time()
    ys_all = []
    vid_paths_all = []
    for iter_, (vid_names, labels, imgs) in enumerate(test_loader):
        model.eval()
        data_time.update(time.time()-end)
        target_var = Variable(labels.cuda(args.gpu_id[0]))

        # forward
        ys, = model([imgs])
        loss = crit(ys, target_var)

        loss_meter.update(loss.data[0], len(imgs))

        prec1, prec5 = accuracy(ys.data.cpu(), labels, topk=(1, 5))

        top1_meter.update(prec1[0], len(imgs))
        top5_meter.update(prec5[0], len(imgs))

        ys = F.softmax(ys)
        ys_all.append(ys.data.cpu().numpy())
        vid_paths_all = vid_paths_all + list(vid_names)

        batch_time.update(time.time() - end)
        end = time.time()

        if iter_ % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                iter_, len(test_loader), batch_time=batch_time, loss=loss_meter,
                top1=top1_meter, top5=top5_meter))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1_meter, top5=top5_meter))

    ys_all = np.concatenate(ys_all, 0)
    if args.resume == '':
        score_name = '{}_{}_score.txt'.format(args.modality, args.eval_split)
    elif args.modality == 'focus_rgb':
        score_name = 'focus_rgb_init_depth_{}_score.txt'.format(args.eval_split)
    elif args.modality == 'focus_depth':
        score_name = 'focus_depth_init_rgb_{}_score.txt'.format(args.eval_split)
    else:
        raise RuntimeError('modality wrong!')

    if is_test:
        save_score(os.path.join('logs', 'test/score', score_name), vid_paths_all, ys_all)
    else:
        save_score(os.path.join('logs', 'train/score', score_name), vid_paths_all, ys_all)
    model.train()
    return top1_meter.avg

def save_score(fn, vid_paths, ys):
    if not os.path.exists(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))

    with open(fn, 'w') as f:
        for vid_path, y in zip(vid_paths, ys):
            f.write('{}'.format(vid_path))
            for y_ in y:
                f.write('\t{}'.format(y_))
            f.write('\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_adjust_learning_rate(optimizer, lr0, step, step_size, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (gamma ** (step // step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
