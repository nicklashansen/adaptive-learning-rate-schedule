import numpy as np
import torch
from copy import deepcopy
import warnings
import setproctitle
import pickle as pkl
import utils

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
            print(f'Warning: agent is tested on {args.dataset} but was trained on {experiment_dataset}.')
    else:
        print(f'Running saved schedule for ALRS testing...\nArgs:\n{utils.args_to_str(args)}\n')

    env = utils.make_alrs_env(args, test=True)

    baseline_args = deepcopy(args)
    baseline_config = utils.load_file_as_dict(args.dataset+'_'+args.architecture, path='data/baselines/')
    baseline_args.initial_lr = baseline_config['initial_lr']
    baseline_env = utils.make_alrs_env(baseline_args, test=True, baseline=True)

    displayed_rendering_error = False

    def run_test(env, baseline_env):
        global displayed_rendering_error

        try:
            model = PPO2.load('data/'+test_id)
            model.set_env(env)
        except:
            raise ValueError('Error: failed to load PPO2 model from path "data/'+test_id+'". Missing?')

        env.reset()
        baseline_env.reset()
        state, baseline_env = env.alrs.reset_and_sync(baseline_env)

        done = False
        best_val_loss = np.inf
        best_val_loss_baseline = np.inf
        global_step = 0
        lr = baseline_env.alrs.lr

        while not done:

            # Take step with auto-learned schedule
            action, _ = model.predict(state)
            state, _, done, _ = env.step(action)

            # Take step with baseline schedule
            action, lr = utils.step_decay_action(lr, global_step, baseline_config['discount_step'], baseline_config['discount_factor'])
            baseline_env.step(action)
            global_step += args.update_freq

            # Save best validation loss
            best_val_loss = min(env.envs[0].info_list[-1]['val_loss'], best_val_loss)
            best_val_loss_baseline = min(baseline_env.envs[0].info_list[-1]['val_loss'], best_val_loss_baseline)

            try:
                env.alrs.render(baseline=baseline_env)
            except:
                if not displayed_rendering_error:
                    displayed_rendering_error = True
                    print('Warning: device does not support rendering.')

        print(f'Val loss: {best_val_loss}\nVal loss (baseline): {best_val_loss_baseline}')

        loss, acc = env.alrs.test()
        print(f'Test loss: {loss}\nTest accuracy: {acc}')

        return best_val_loss

    while True:
        run_test(env, baseline_env)

    print('Testing terminated successfully!')
    