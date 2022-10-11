# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from models import resnet18, resnet18_bank
from utils.tools import (create_logger,
                         restart_from_checkpoint,
                         AverageMeter, PaceAverageMeter,
                         SingleLoopSampler, gen_cls_inds)
from utils.hsic import RbfHSICB

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 8000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Trains a IN1K Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['in1k'],
                    help='Choose between IN1K.')
parser.add_argument('--model', '-m', type=str, default='r18',
                    choices=['r18', 'r18_bank'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.2, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=512, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Conditional-i parameters
parser.add_argument('--hsic-scale', default=16384, type=float, help='hsic loss scale')
parser.add_argument('--hsic-sigma', default=3., type=float, help='hsic sigma')
parser.add_argument('--cond-i-weight', default=0., type=float, help='hsic cls loss weight')
parser.add_argument('--cond-i-warmup', default=5, type=int, help='warmup epoch of hsic cls loss')
parser.add_argument('--bank_size', default=512, type=int, help='bank size')
parser.add_argument('--sample-cls', default=0, type=int, help='number of sampling class')
parser.add_argument('--bank_cls_num', default=0, type=int, help='bank_cls_num')
parser.add_argument('--shuffle-cls', default=0, type=int, help='whether shuffle cls')
# Checkpoints
parser.add_argument('--code_dir', type=str, default=None, help='Folder to save checkpoints.')
parser.add_argument('--save', '-s', type=str, default='./snapshots/oe_scratch', help='Folder to save checkpoints.')
parser.add_argument('--resume', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
parser.add_argument('--ckp-freq', type=int, default=10, help='Save the model periodically')
parser.add_argument('--log-freq', type=int, default=100, help='Save the log periodically')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=16, help='Pre-fetching threads.')
# Random seed
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--disable_random", type=int, default=1, help="disable_random")
parser.add_argument("--shuffle-ood", type=int, default=1, help="shuffle ood training data.")


def main():
    args = parser.parse_args()

    if args.code_dir is None:
        args.code_dir = path.dirname(path.abspath(__file__))

    # Make save directory
    os.makedirs(os.path.join(args.code_dir, args.save, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.code_dir, args.save, 'checkpoints'), exist_ok=True)

    state = {k: v for k, v in args._get_kwargs()}

    # create logger and tensorboard writer
    global logger
    logger = create_logger(
        os.path.join(args.code_dir, args.save, 'logs', 'train.log'), 0)

    # set seed
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # mean and standard deviation of channels of IN1K images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Create training and val dataset
    train_transform = trn.Compose([
        trn.Resize(64),
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(64, padding=8),
        trn.ToTensor(),
        trn.Normalize(mean, std)])

    test_transform = trn.Compose([
        trn.Resize(64),
        trn.CenterCrop(64),
        trn.ToTensor(),
        trn.Normalize(mean, std)])

    train_data_in = dset.ImageFolder(
        os.path.join(args.code_dir, 'data/in1k/in1k_train_64'),
        transform=train_transform)
    test_data = dset.ImageFolder(
        os.path.join(args.code_dir, 'data/in1k/in1k_val_64'),
        transform=test_transform)

    num_classes = 1000
    args.num_classes = num_classes
    if args.bank_cls_num == 0:
        args.bank_cls_num = num_classes
    else:
        assert args.sample_cls > 0
    logger.info(f"=> Load inlier data ({args.dataset}): {len(train_data_in)} images")

    # Create ood dataset
    ood_data = dset.ImageFolder(
        os.path.join(args.code_dir, 'data/in21k/train'),
        transform=trn.Compose([
            trn.Resize(64),
            trn.RandomCrop(64, padding=8),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std)]))
    logger.info(f"=> Load outlier data (IN21K): {len(ood_data)} images")

    # Create dataloader
    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    if args.shuffle_ood:
        ood_sampler = SingleLoopSampler(
            ood_data, len(train_data_in) * int(args.oe_batch_size / args.batch_size),
            shuffle=True, seed=0)
    else:
        ood_sampler = None
    train_loader_ood = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.oe_batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, sampler=ood_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    # Create model
    logger.info("=> creating model '{}'".format(args.model))
    if args.model == 'r18':
        net = resnet18(
            num_classes=num_classes)
    elif args.model == 'r18_bank':
        net = resnet18_bank(
            args.oe_batch_size,
            num_classes=num_classes)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True  # fire on all cylinders

    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    # Restore model if desired
    to_restore = {"epoch": 0}
    resume_path = os.path.join(args.code_dir, args.save, "checkpoint.pth.tar")
    if args.resume is not None:
        resume_path = os.path.join(args.code_dir, args.resume)
    restart_from_checkpoint(
        resume_path,
        run_variables=to_restore,
        state_dict=net,
        optimizer=optimizer,
        scheduler=scheduler)
    start_epoch = to_restore["epoch"]

    logger.info('=> Beginning training')

    # Main loop
    for epoch in range(start_epoch, args.epochs):
        state['epoch'] = epoch

        train(train_loader_in, train_loader_ood, net, optimizer, scheduler, state, args)
        test(test_loader, net, state)

        # Save model
        save_dict = {
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(save_dict, os.path.join(args.code_dir, args.save, "checkpoint.pth.tar"))
        if (epoch + 1) % args.ckp_freq == 0 or (epoch + 1) == args.epochs:
            shutil.copyfile(
                os.path.join(args.code_dir, args.save, "checkpoint.pth.tar"),
                os.path.join(args.code_dir, args.save, 'checkpoints', "ckp-" + str(epoch) + ".pth"),
            )


# train function
def train(train_loader_in, train_loader_ood, net, optimizer, scheduler, state, args):
    net.train()  # enter train mode
    epoch = state['epoch']
    sample_number = len(train_loader_in)

    left_time = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = PaceAverageMeter(pace=200)
    hsic_rbf_bs = RbfHSICB(sigma_x=args.hsic_sigma)
    if args.cond_i_weight > 0.:
        cond_i_losses = PaceAverageMeter(pace=200)
    losses = PaceAverageMeter(pace=200)
    top1 = PaceAverageMeter(pace=200)

    if args.disable_random:
        # fixed version for removing randomness of training ood data
        train_loader_ood.dataset.offset = int(len(train_loader_ood.dataset) * 1. / args.epochs * epoch)
    else:
        # OE random version
        train_loader_ood.dataset.offset = np.random.randint(len(train_loader_ood.dataset))

    end = time.time()
    for it, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_ood)):
        data_time.update(time.time() - end)
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        in_bs = target.size(0)
        total_it = epoch * sample_number + it

        data, target = data.cuda(), target.cuda()

        # forward
        x, feat = net(data, return_feat=True)
        in_feat = feat[:in_bs]
        ood_feat = feat[in_bs:]

        # get cls bank
        if 'bank' in args.model:
            queue = net.queue.clone().detach()  # (num_classes, K, D)
            with torch.no_grad():
                net.dequeue_and_enqueue(in_feat, target)

        # cross entropy loss
        ce_loss = F.cross_entropy(x[:in_bs], target)
        loss = ce_loss
        ce_losses.update(ce_loss.item(), data.size(0))

        # Conditional independence loss: Eq. (6)
        if args.cond_i_weight > 0.:
            ratio = (it + epoch * sample_number) / (args.cond_i_warmup * sample_number)
            w = args.hsic_scale * args.cond_i_weight
            if epoch < args.cond_i_warmup:
                w *= ratio
            cond_i_loss = ood_feat.sum() * 0.
            assert args.oe_batch_size % args.bank_size == 0
            _ood_feat = ood_feat.reshape(-1, args.bank_size, ood_feat.size(-1))[None, ...]
            _ood_feat = _ood_feat.repeat(args.num_classes, 1, 1, 1)
            _queue = queue[:, None, ...].repeat(1, args.oe_batch_size // args.bank_size, 1, 1)
            if args.sample_cls > 0:
                cls_inds = gen_cls_inds(total_it, args)
                _ood_feat = _ood_feat[cls_inds, ...]
                _queue = _queue[cls_inds, ...]
            _ood_feat = _ood_feat.reshape(-1, args.bank_size, ood_feat.size(-1))
            _queue = _queue.reshape(-1, args.bank_size, queue.size(-1))
            cond_i_loss += w * hsic_rbf_bs(_ood_feat, _queue).mean()
            cond_i_losses.update(cond_i_loss.item(), data.size(0))
            loss += cond_i_loss

        acc1, acc5 = accuracy(x[:len(in_set[0]), :args.num_classes], target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        left_time.update(batch_time.val * ((args.epochs - epoch) * len(train_loader_in) - it) / 3600.0)
        end = time.time()

        if (it + 1) % args.log_freq == 0:
            log_str = ''
            log_str += f"Epoch: [{epoch}][{it + 1}/{sample_number}]\t"
            log_str += f"Left Time {left_time.val:.3f} ({left_time.avg:.3f})\t"
            log_str += f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
            log_str += f"CE Loss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t"
            if args.cond_i_weight > 0.:
                log_str += f"Conditional Independence Loss {cond_i_losses.val:.4f} ({cond_i_losses.avg:.4f})\t"
            log_str += f"Error@1 {(1. - top1.val) * 100:.4f} ({(1. - top1.avg) * 100:.4f})\t"
            log_str += f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            log_str += f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
            log_str += f"Lr: {optimizer.param_groups[0]['lr']:.4f}"
            logger.info(log_str)


# test function
def test(test_loader, net, state):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    logger.info(f'Eval: * Error@1 {(1. - test_accuracy) * 100.:.3f}')


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
