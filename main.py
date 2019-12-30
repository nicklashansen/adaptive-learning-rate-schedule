import gym
import numpy as np
import torch
import os
import warnings
import setproctitle

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils

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
    print(f'Running PPO2 controller for ALRS training...\nArgs:\n{utils.args_to_str(args)}\n')

    data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)

    if args.dataset == 'mnist':
        net_fn = lambda: networks.MLP(784, 256, 128, 10)
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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.ppo2_gamma)
    
    env_seeds = [0, 1, 2, 3]
    for i in  range(env.num_envs):
        env.venv.envs[i].set_random_state(env_seeds[i])

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
            #'net_arch': [64, {'pi': [32], 'vf': [32]}],
            'cnn_extractor': None
        },
        tensorboard_log='data/tensorboard/ppo2_alrs'
    )

    best_episode_reward = -np.inf

    def callback(_locals, _globals):
        """
        Callback called every n steps.
        """
        global best_episode_reward, model

        if model.episode_reward > best_episode_reward:
            print(f'Achieved new maximum reward: {float(model.episode_reward)} (previous: {float(best_episode_reward)})')
            best_episode_reward = float(model.episode_reward)
            model.save('data/ppo2_alrs')

        return True

    model.learn(
        total_timesteps=args.ppo2_total_timesteps,
        tb_log_name=utils.args_to_str(args, separate_lines=False),
        reset_num_timesteps=False,
        callback=callback
    )

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

    model = PPO2.load('data/ppo2_alrs')
    test(model, env)
    print('Training terminated successfully!')