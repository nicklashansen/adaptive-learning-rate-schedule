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
    experiment_id = utils.get_random_string()
    setproctitle.setproctitle('PPO2-ALRS-'+experiment_id.upper())
    print(f'Running PPO2 controller for ALRS training...\nArgs:\n{utils.args_to_str(args)}\n')

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

    model = PPO2(
        policy=MlpPolicy,
        env=env,
        gamma=args.ppo2_gamma,
        n_steps=args.ppo2_update_freq,
        learning_rate=args.ppo2_lr,
        nminibatches=1,
        verbose=1,
        policy_kwargs={
            'act_fun': tf.nn.relu,
            'net_arch': [{'pi': [64, 32], 'vf': [64, 32]}]
        },
        tensorboard_log='data/tensorboard/ppo2_alrs'
    )

    utils.args_to_file(args, experiment_id)
    best_episode_reward = -np.inf

    def callback(_locals, _globals):
        """
        Callback called every n steps.
        """
        global experiment_id, best_episode_reward, model, args

        if model.episode_reward > best_episode_reward:
            print(f'Achieved new maximum reward: {float(model.episode_reward)} (previous: {float(best_episode_reward)})')
            best_episode_reward = float(model.episode_reward)
            model.save('data/'+experiment_id)
        
        save_interval = 50000 if args.dataset == 'mnist' else 10000
        if model.num_timesteps % save_interval == 0 and model.num_timesteps > 0:
            model.save('data/'+experiment_id+'_'+str(model.num_timesteps))

        return True

    model.learn(
        total_timesteps=args.ppo2_total_timesteps,
        tb_log_name=experiment_id,
        reset_num_timesteps=False,
        callback=callback
    )
    print('Training terminated successfully!')
