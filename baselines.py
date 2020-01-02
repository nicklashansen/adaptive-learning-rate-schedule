import gym
import numpy as np
import torch
import os
import warnings
import setproctitle
from torchvision.models.resnet import resnet18
from lenet import LeNet5

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils


if __name__ == '__main__':
    args = utils.parse_args()
    setproctitle.setproctitle('PPO2-ALRS')
    print(f'Running baseline methods for ALRS testing...\nArgs:\n{utils.args_to_str(args)}\n')

    data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)

    if args.dataset == 'mnist':
        data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: networks.MLP(784, 256, 128, 10)

    elif args.dataset == 'cifar10':
        data = utils.load_cifar10(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: resnet18(num_classes=10)

    elif args.dataset == 'fa-mnist':
        data = utils.load_fashion_mnist(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: LeNet5(num_channels_in=1, num_classes=10, img_dims=(28, 28))

    env = AdaptiveLearningRateOptimizer(
        train_dataset=data[0],
        val_dataset=data[1],
        net_fn=net_fn,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        num_train_steps=args.num_train_steps,
        initial_lr=args.initial_lr,
        num_devices=args.num_devices,
        discrete=args.discrete,
        verbose=False
    )

    def run_baseline(env, mode):
        """
        Constant (initial LR)
        """
        env.reset()
        done = False
        step_count = 0

        while not done:

            action = 2

            if mode == 'step_decay':
                if step_count == 50:
                    action = 0
                elif step_count in {400, 800}:
                    action = 1

            _, _, done, _ = env.step(action)
            step_count += args.update_freq

            try:
                env.render()
            except:
                print('Warning: device does not support rendering. Skipping...')

        utils.save_baseline(env.info_list, mode)

    run_baseline(env, 'initial_lr')
    run_baseline(env, 'step_decay')

    print('Testing terminated successfully!')
    