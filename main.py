import gym
import numpy as np
import torch
import os
import warnings
import setproctitle

from smallrl import algorithms, environments, networks, demos
from environment import AdaptiveLearningRateOptimizer
import utils
from torchvision.models.resnet import resnet18
from lenet import LeNet5

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy
    from stable_baselines.common import make_vec_env, schedules
    from stable_baselines.common.vec_env import VecNormalize
    from stable_baselines import PPO2


if __name__ == '__main__':
    args = utils.parse_args()
    experiment_id = utils.get_random_string()
    setproctitle.setproctitle('PPO2-ALRS-'+experiment_id.upper())
    print(f'Running PPO2 controller for ALRS training...\nArgs:\n{utils.args_to_str(args)}\n')
    print(f'Experiment ID:', experiment_id)

    data, net_fn = utils.load_dataset_and_network(dataset=args.dataset)

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
            'action_range': args.action_range,
            'verbose': False
        }
    )
    env = VecNormalize(
        venv=env,
        norm_obs=args.ppo2_norm_obs,
        norm_reward=args.ppo2_norm_reward,
        clip_obs=args.ppo2_cliprange if args.ppo2_cliprange > 0 else 10,
        clip_reward=args.ppo2_cliprange if args.ppo2_cliprange > 0 else 10,
        gamma=args.ppo2_gamma
    )

    lr_schedule = schedules.LinearSchedule(
        schedule_timesteps=args.ppo2_total_timesteps,
        initial_p=args.ppo2_lr,
        final_p=args.ppo2_lr*0.1
    )

    ent_coef_schedule = schedules.LinearSchedule(
        schedule_timesteps=args.ppo2_total_timesteps,
        initial_p=args.ppo2_ent_coef,
        final_p=args.ppo2_ent_coef*0.01
    )

    model = PPO2(
        policy=MlpPolicy,
        env=env,
        gamma=args.ppo2_gamma,
        n_steps=args.ppo2_update_freq,
        ent_coef=args.ppo2_ent_coef,
        learning_rate=lr_schedule.value,
        nminibatches=args.ppo2_nminibatches,
        noptepochs=args.ppo2_noptepochs,
        cliprange=args.ppo2_cliprange,
        verbose=1,
        policy_kwargs={
            'act_fun': tf.nn.relu,
            'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}]
        },
        tensorboard_log='data/tensorboard/ppo2_alrs'
    )

    utils.args_to_file(args, experiment_id)
    best_episode_reward = -np.inf
    best_val_loss = np.inf

    def callback(_locals, _globals):
        """
        Callback called every n steps.
        """
        global experiment_id, best_episode_reward, best_val_loss, model, ent_coef_schedule, args

        minor_save_interval = 2500  if args.dataset == 'mnist' else 500
        major_save_interval = 25000 if args.dataset == 'mnist' else 5000

        model.ent_coef = ent_coef_schedule.value(model.num_timesteps)

        if model.episode_reward > best_episode_reward:
            print(f'Achieved new maximum reward: {float(model.episode_reward)} (previous: {float(best_episode_reward)})')
            best_episode_reward = float(model.episode_reward)
            model.save('data/'+experiment_id+'_bestreward')

        val_loss = model.env.venv.envs[0].env.info_list[-1]['val_loss']
        if val_loss < best_val_loss and best_val_loss < 1:
            print(f'Achieved new minimum val loss: {float(val_loss)} (previous: {float(best_val_loss)})')
            best_val_loss = float(val_loss)
            model.save(f'data/{experiment_id}_bestval={np.around(best_val_loss, decimals=4)}.zip')
        
        if model.num_timesteps % minor_save_interval == 0 and model.num_timesteps > 0:
            model.save('data/'+experiment_id+'_current')
        
        if model.num_timesteps % major_save_interval == 0 and model.num_timesteps > 0:
            model.save('data/'+experiment_id+'_'+str(model.num_timesteps/100)+'k')

        return True

    tb_log_name = experiment_id
    if args.tb_suffix is not None:
        tb_log_name += '__'+args.tb_suffix

    model.learn(
        total_timesteps=args.ppo2_total_timesteps,
        tb_log_name=tb_log_name,
        reset_num_timesteps=False,
        callback=callback
    )
    print('Training terminated successfully!')
