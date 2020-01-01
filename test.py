import gym
import numpy as np
import torch
import os
import warnings
import setproctitle

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils
from lenet import LeNet5

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines.common.vec_env import VecNormalize
    from stable_baselines import PPO2


if __name__ == '__main__':
    args = utils.parse_args()
    setproctitle.setproctitle('PPO2-ALRS')
    print(f'Running PPO2 controller for ALRS testing...\nArgs:\n{utils.args_to_str(args)}\n')

    if args.dataset == 'mnist':
        data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: networks.MLP(784, 256, 128, 10)

    elif args.dataset == 'cifar10':
        data = utils.load_cifar(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: LeNet5(num_classes=10)

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
            'num_devices': args.num_devices,
            'verbose': False
        }
    )
    env = VecNormalize(env, norm_obs=args.ppo2_norm_obs, norm_reward=args.ppo2_norm_reward, gamma=args.ppo2_gamma)


    def test(model, env):
        model.set_env(env)
        state = env.reset()
        done = False

        while not done:
            action, _ = model.predict(state)
            state, _, done, _ = env.step(action)
            try:
                env.render()
            except:
                print('Warning: device does not support rendering. Skipping...')

    model = PPO2.load('data/iezzgq')
    test(model, env)
    print('Testing terminated successfully!')
    