import numpy as np
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

    best_overall_val_loss = np.inf
    displayed_rendering_error = False

    def run_test(env):
        global best_overall_val_loss, displayed_rendering_error

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
            
            best_val_loss = min(env.envs[0].info_list[-1]['val_loss'], best_val_loss)

            try:
                env.render()
            except:
                if not displayed_rendering_error:
                    displayed_rendering_error = True
                    print('Warning: device does not support rendering.')

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
    