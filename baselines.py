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

    data, net_fn = utils.load_dataset_and_network(
        dataset=args.dataset,
        num_train=args.num_train,
        num_val=args.num_val
    )

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
        lr_noise=False,
        verbose=False
    )

    displayed_rendering_error = False

    def run_baseline(env, mode):
        """
        Constant (initial LR)
        """
        global displayed_rendering_error

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
                if not displayed_rendering_error:
                    displayed_rendering_error = True
                    print('Warning: device does not support rendering.')

        utils.save_baseline(env.info_list, mode+'_'+args.dataset)

    for baseline in ['initial_lr', 'step_decay']:
        print(f'Running {baseline} baseline...')
        run_baseline(env, baseline)
        print('Done!')

    print('Testing terminated successfully!')
    