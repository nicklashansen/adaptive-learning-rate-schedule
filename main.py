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


if __name__ == '__main__':
    args = utils.parse_args()
    experiment_id = utils.get_random_string()
    setproctitle.setproctitle('PPO2-ALRS-'+experiment_id.upper())
    print(f'Running PPO2 controller for ALRS training...\nArgs:\n{utils.args_to_str(args)}\n')
    print(f'Experiment ID:', experiment_id)

    env = utils.make_alrs_env(args)
    model = utils.make_ppo2_controller(env, args)

    utils.args_to_file(args, experiment_id)
    best_episode_reward = -np.inf
    best_val_loss = np.inf

    def callback(_locals, _globals):
        """
        Callback called every n steps.
        """
        global experiment_id, best_episode_reward, best_val_loss, model, args

        minor_save_interval = 2000  if args.dataset == 'mnist' else 1000
        major_save_interval = 10000 if args.dataset == 'mnist' else 5000

        steps = str(int(model.num_timesteps/100))+'k'
        val_loss = model.env.venv.envs[0].env.latest_end_val

        print('Val loss:', val_loss)

        if val_loss is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=val_loss)])
            _locals['writer'].add_summary(summary, model.num_timesteps)
            if val_loss < best_val_loss:
                print(f'Achieved new minimum val loss: {val_loss} (previous: {best_val_loss})')
                best_val_loss = val_loss
                model.save(f'data/{experiment_id}_steps={steps}_val={str(np.around(best_val_loss, decimals=4))}.zip')

        if model.num_timesteps % minor_save_interval == 0 and model.num_timesteps > 0:
            model.save(f'data/{experiment_id}_current')
        
        if model.num_timesteps % major_save_interval == 0 and model.num_timesteps > 0:
            model.save(f'data/{experiment_id}_steps={steps}.zip')

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
