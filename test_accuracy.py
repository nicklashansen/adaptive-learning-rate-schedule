import numpy as np
import warnings
import setproctitle
import pickle as pkl
import utils

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    args = utils.parse_args()
    setproctitle.setproctitle('PPO2-ALRS')

    env = utils.make_alrs_env(args, test=True)
    baseline = utils.values_from_list_of_dicts(utils.load_baseline(args.dataset), 'lr')

    test_loss, test_acc = utils.AvgLoss(), utils.AvgLoss()
    num_runs = 10

    for run in range(num_runs):

        env.reset()
        done = False
        step = 0
        alrs = env.venv.envs[0].env

        while not done:
            
            # TODO: Fix baseline lr rates, they are inaccurate (numerically unstable?)

            action = np.array(baseline[step] / alrs.lr).reshape(1,)
            _, _, done, _ = env.step(action)
            step += 1

            if done:
                loss, acc = alrs.test()
                test_loss += loss
                test_acc += acc

    results = {
        'num_runs': num_runs,
        'dataset': args.dataset,
        'avg_loss': test_loss.avg,
        'avg_acc': test_acc.avg
    }
    print(f'Results:\n{results}')
    utils.dict_to_file(results, 'test_'+args.dataset, path='results/')
