import gym
import numpy as np
import torch
import os
import warnings
import setproctitle

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils


if __name__ == '__main__':
    args = utils.parse_args()
    setproctitle.setproctitle('PPO2-ALRS')
    print(f'Running baseline methods for ALRS testing...\nArgs:\n{utils.args_to_str(args)}\n')

    data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)

    if args.dataset == 'mnist':
        net_fn = lambda: networks.MLP(784, 256, 128, 10)
    elif args.dataset == 'cifar10':
        net_fn = lambda: networks.CNN_MLP(channels=(3,16,8,4), kernel_sizes=(5,5,5), paddings=(0,0,0), sizes=(1600,10))

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
    