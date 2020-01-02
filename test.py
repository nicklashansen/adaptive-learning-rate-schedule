import gym
import numpy as np
import torch
import os
import warnings
import setproctitle
import pickle as pkl
from torchvision.models.resnet import resnet18
from lenet import LeNet5

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

    test_id = args.test_id if args.test_schedule == 'none' else None
    test_schedule = args.test_schedule if args.test_schedule != 'none' else None

    if test_id is not None:
        try:
            exp_id = args.test_id.split('_')[0] if '_' in args.test_id else args.test_id
            experiment_args = utils.load_args_file_as_dict(exp_id)
            print(f'Running PPO2 controller for ALRS testing...\nTrained with args:\n{utils.args_to_str(experiment_args)}\n')
            print(f'Experiment ID:', args.test_id)
        except:
            raise ValueError(f'Experiment with id {args.test_id} could not be found!')

        experiment_dataset = experiment_args['dataset']
        if args.dataset != experiment_dataset:
            raise Warning(f'Agent is tested on {args.dataset} but was trained on {experiment_dataset}.')
    
    else:
        print(f'Running saved schedule for ALRS testing...\nArgs:\n{utils.args_to_str(args)}\n')

    if args.dataset == 'mnist':
        data = utils.load_mnist(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: networks.MLP(784, 256, 128, 10)

    elif args.dataset == 'cifar10':
        data = utils.load_cifar10(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: resnet18(num_classes=10)

    elif args.dataset == 'fa-mnist':
        data = utils.load_fashion_mnist(num_train=args.num_train, num_val=args.num_val)
        net_fn = lambda: LeNet5(num_channels_in=1, num_classes=10, img_dims=(28, 28))

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
            'discrete': args.discrete,
            'verbose': False
        }
    )
    env = VecNormalize(
        venv=env,
        norm_obs=args.ppo2_norm_obs,
        norm_reward=args.ppo2_norm_reward,
        gamma=args.ppo2_gamma
    )

    best_overall_val_loss = np.inf

    def run_test(env):
        global best_overall_val_loss

        if test_id is not None:
            model = PPO2.load('data/'+test_id)
            model.set_env(env)
            actions = []
        else:
            with open('results/'+test_schedule+'.pkl', 'rb') as f:
                actions = pkl.load(f)

        state = env.reset()
        done = False
        best_val_loss = np.inf

        while not done:
            if test_id is not None:
                action, _ = model.predict(state)
                state, _, done, _ = env.step(action)
                actions.append(action)
            else:
                action = actions.pop(0) if len(actions) > 0 else 2
                env.step(action)
            
            best_val_loss = min(env.venv.envs[0].info_list[-1]['val_loss'], best_val_loss)

            try:
                env.render()
            except:
                print('Warning: device does not support rendering. Skipping...')

        best_overall_val_loss = min(best_val_loss, best_overall_val_loss)

        if test_id is not None and best_overall_val_loss == best_val_loss:
            with open('results/experiment.pkl', 'wb') as f:
                pkl.dump(actions, f, protocol=pkl.HIGHEST_PROTOCOL)

        print(f'Achieved a validation loss of {best_val_loss} (best: {best_overall_val_loss})')

        return best_val_loss

    if test_id is not None:
        while True:
            run_test(env)
    else:
        run_test(env)

    print('Testing terminated successfully!')
    