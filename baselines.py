import numpy as np
import torch

import utils


if __name__ == '__main__':
    args = utils.parse_args()
    print(f'Running baseline for ALRS testing...\nArgs:\n{utils.args_to_str(args)}\n')
    
    displayed_rendering_error = False

    best_config = None
    best_info_list = None
    best_val_loss = np.inf

    initial_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    discount_steps = [10, 20, 50, 100]
    discount_factors = [.99, .9, .88]

    for initial_lr in initial_lrs:
        for discount_step in discount_steps:
            for discount_factor in discount_factors:
                
                print(f'Initial LR: {initial_lr}\nDiscount step: {discount_step}\nDiscount factor: {discount_factor}')

                args.initial_lr = initial_lr
                env = utils.make_alrs_env(args, test=True, baseline=True)

                env.reset()
                done = False
                global_step = 0
                current_lr = initial_lr
                info_list = []

                while not done:
                    action, new_lr = utils.step_decay_action(current_lr, global_step, discount_step, discount_factor)

                    _, _, done, info = env.step(action)
                    global_step += args.update_freq
                    current_lr = new_lr
                    info_list.append(info)

                    try:
                        env.render()
                    except:
                        if not displayed_rendering_error:
                            displayed_rendering_error = True
                            print('Warning: device does not support rendering.')
                
                val_loss = env.venv.envs[0].env.latest_end_val
                print('Final validation loss:', val_loss)

                if val_loss < best_val_loss:
                    best_config = {
                        'dataset': args.dataset,
                        'architecture': args.architecture,
                        'initial_lr': initial_lr,
                        'discount_step': discount_step,
                        'discount_factor': discount_factor,
                        'val_loss': val_loss,
                        'log_val_loss': np.log(val_loss)
                    }
                    best_info_list = info_list
                    best_val_loss = val_loss

    print(f'Found best configuration:\n{best_config}')
    filename = args.dataset+'_'+args.architecture
    utils.dict_to_file(best_config, filename, path='data/baselines/')
    utils.save_baseline(best_info_list, filename)
