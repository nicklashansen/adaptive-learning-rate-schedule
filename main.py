import gym
import numpy as np
import torch

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


if __name__ == '__main__':
    data = utils.load_mnist(num_train=10000, num_val=1000)
    net_fn = lambda: networks.MLP(784, 64, 10)
    print(f'CUDA is available: {torch.cuda.is_available()}')
    env = make_vec_env(
        env_id=AdaptiveLearningRateOptimizer,
        n_envs=4,
        env_kwargs={
            'train_dataset': data[0],
            'val_dataset': data[1],
            'net_fn': net_fn,
            'batch_size': 1000,
            'update_freq': 10,
            'num_train_steps': 1000,
            'initial_lr': 1e-3,
            'num_devices': 4 if torch.cuda.is_available() else 'cpu'
        }
    )

    model = PPO2(MlpPolicy, env, n_steps=50, learning_rate=1e-3, verbose=1)
    model.learn(total_timesteps=100000)
    model.save('ppo2_alrs')


"""
if __name__ == '__main__':
    #data = utils.load_cifar(num_train=10000, num_val=1000)
    data = utils.load_mnist(num_train=10000, num_val=1000)

    #net_fn = lambda: networks.CNN_MLP(channels=(3,16,8,4), kernel_sizes=(5,5,5), paddings=(0,0,0), sizes=(1600,10))
    net_fn = lambda: networks.MLP(784, 64, 10)

    env = AdaptiveLearningRateOptimizer(
        train_dataset=data[0],
        val_dataset=data[1],
        net_fn=net_fn,
        batch_size=1000,
        update_freq=10,
        num_train_steps=1000,
        initial_lr=1e-3,
        device=1 if torch.cuda.is_available() else 'cpu'
    )
    env = environments.TorchEnv(env)

    obs_dims, action_dims = environments.get_env_dimensions(env)
    net = networks.MLP(obs_dims, 32, action_dims)
    agent = algorithms.REINFORCE(
        env=env,
        net=net,
        optimizer=torch.optim.Adam(net.parameters(), lr=5e-3),
        discount_factor=1,
        entropy_weight=0
    )

    demos.train_agent(
        agent=agent,
        criterion={'n': 1, 'target_reward': np.inf, 'max_episodes': 1000},
        verbose=True
    )
"""