import gym
import numpy as np
import torch
import os
import warnings  

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines import PPO2


if __name__ == '__main__':
    args = utils.parse_args()
    print(f'Running PPO2 controller for ALRS training...\nArgs:\n{args}\n')

    data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)

    if args.dataset == 'mnist':
        net_fn = lambda: networks.MLP(784, 64, 10)
    elif args.dataset == 'cifar10':
        net_fn = lambda: networks.CNN_MLP(channels=(3,16,8,4), kernel_sizes=(5,5,5), paddings=(0,0,0), sizes=(1600,10))

    env = make_vec_env(
        env_id=AdaptiveLearningRateOptimizer,
        n_envs=args.num_devices,
        env_kwargs={
            'train_dataset': data[0],
            'val_dataset': data[1],
            'net_fn': net_fn,
            'batch_size': args.batch_size,
            'update_freq': args.update_freq,
            'num_train_steps': args.num_train_steps,
            'initial_lr': args.initial_lr,
            'num_devices': args.num_devices
        }
    )

    model = PPO2(
        policy=MlpPolicy,
        env=env,
        gamma=args.ppo2_gamma,
        n_steps=args.ppo2_update_freq,
        learning_rate=args.ppo2_lr,
        verbose=1
    )
    model.learn(total_timesteps=args.ppo2_total_timesteps)
    model.save('ppo2_alrs')
