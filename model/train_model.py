#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os
from PIL import Image

from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import argparse

import generator
import discriminator

from rough2lineDataset import Rough2LineDataset
from training_visualizer import test_samples_simplification

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./images/',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--test_visual_interval', type=int, default=10000,
                        help='Interval of drawing test images')
    parser.add_argument('--test_out', default='test/',
                        help='DIrectory to output test samples')
    parser.add_argument('--test_image_path', default='./test_samples/',
                        help='Directory of image files for testing')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    root = args.dataset
    #model = "./model_paint"

    gen = generator.GEN()
    #serializers.load_npz("result/model_iter_10000", gen)

    dis = discriminator.DIS()
    #serializers.load_npz("result/model_dis_iter_20000", dis)

    dataset = Rough2LineDataset(
        "dat/rough_line_train.dat", root + "rough/", root + "line/", train=True)

    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()  # Copy the model to the GPU

    # Setup optimizer parameters.
    opt = optimizers.Adam(alpha=0.0001)
    opt.setup(gen)
    opt.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_gen')

    opt_d = chainer.optimizers.Adam(alpha=0.0001)
    opt_d.setup(dis)
    opt_d.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')

    # Set up a trainer
    updater = ganUpdater(
        models=(gen, dis),
        iterator={
            'main': train_iter,
            #'test': test_iter
        },
        optimizer={
            'gen': opt,
            'dis': opt_d},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    snapshot_interval2 = (args.snapshot_interval * 2, 'iteration')
    trainer.extend(extensions.dump_graph('gen/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval2)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'gen_dis_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'gen/loss', 'gen/loss_L', 'gen/loss_adv', 'dis/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(test_samples_simplification(updater, gen, args.test_out, args.test_image_path),
                   trigger=(args.test_visual_interval, 'iteration'))

    trainer.run()

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(out_dir, 'model_final'), gen)
    chainer.serializers.save_npz(os.path.join(out_dir, 'optimizer_final'), opt)


class ganUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        super(ganUpdater, self).__init__(*args, **kwargs)

        # 0 for dataset
        # 1 for fake
        # G_out: output of Generator
        # gt: ground truth
    def loss_gen(self, gen, G_out, gt, alpha=0.00008):
        loss_L = F.mean_squared_error(G_out, gt)
        loss_adv = F.softmax_cross_entropy(self.dis(G_out), Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss = loss_L + alpha * loss_adv
        chainer.report({'loss': loss, "loss_L": loss_L, 'loss_adv': loss_adv}, gen)
        return loss

    def loss_dis(self, dis, G_out, gt, alpha=0.00008):
        loss_fake = F.softmax_cross_entropy(self.dis(G_out), Variable(xp.ones(batchsize, dtype=np.int32)))
        loss_real = F.softmax_cross_entropy(self.dis(gt), Variable(xp.zeros(batchsize, dtype=np.int32)))
        loss = alpha * (loss_fake + loss_real)
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        w_in = 384
        w_out = 384

        x_in = xp.zeros((batchsize, 1, w_in, w_in)).astype("f")
        gt = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")

        for i in range(batchsize):
            x_in[i, :] = xp.asarray(batch[i][0])
            gt[i, :] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        gt = Variable(gt)

        G_out = self.gen(x_in, test=False)

        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen_optimizer.update(self.loss_gen, self.gen, G_out, gt)
        G_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, self.dis, G_out, gt)

if __name__ == '__main__':
    main()