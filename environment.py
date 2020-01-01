import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import setproctitle
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns

import utils
from smallrl.utils import smooth


class AdaptiveLearningRateOptimizer(gym.Env):
    """
    Optimization environment that implements the gym environment interface.
    Can be used to learn an adaptive learning rate schedule.

    Observations (6):
        0: Log training loss
        1: Log validation loss
        2: Log variance of predictions
        3: Mean of output weight matrix
        4: Variance of output weight matrix
        5: Learning rate

    Actions (3):
        0: Doubles the LR
        1: Halves the LR
        2: No-op
    """
    def __init__(self, train_dataset, val_dataset, net_fn, batch_size, update_freq, num_train_steps, initial_lr, num_devices, verbose=False):
        super().__init__()

        class SpecDummy():
            def __init__(self, id):
                self.id = id
        
        self.spec = SpecDummy(id='AdaptiveLearningRate-v0')
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.net_fn = net_fn
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.num_train_steps = num_train_steps
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32
            )
        self.initial_lr = initial_lr
        self.num_devices = num_devices
        self.verbose = verbose
        self.info_list = []
        self.seed(0)
        self.cuda = torch.cuda.is_available()
        assert num_devices == 1 or self.cuda

    
    def seed(self, seed):
        """
        Sets the internal random state of the environment.
        """
        self.random_state = np.random.RandomState(seed)

    
    def _update_lr(self, action, clip=True):
        """
        Updates the current learning rate according to a given action.
        Optionally clips to [1e-6, 1] range.
        """
        if action == 0:
            self.lr *= 2
        elif action == 1:
            self.lr /= 2
        if clip:
            self.lr = np.clip(self.lr, 1e-6, 1)
        if action != 2 and self.training_steps != 0:
            self.schedule.step()


    def step(self, action):
        """
        Takes a step in the environment and computes a new state.
        """
        self._update_lr(action)
        train_loss = utils.AvgLoss()
        val_loss = utils.AvgLoss()

        for _ in range(self.update_freq):
            if self.training_steps % self.num_train_batches == 0:
                self.train_iter = iter(self.train_generator)

            x, y = next(self.train_iter)
            if self.cuda:
                with torch.cuda.device(self.device):
                    x = x.cuda()
                    y = y.cuda()
            loss = F.cross_entropy(self.net(x), y)
            train_loss += loss
            self.training_steps += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        yhat_var = utils.AvgLoss()

        for x, y in self.val_generator:
            with torch.no_grad():
                if self.cuda:
                    with torch.cuda.device(self.device):
                        x = x.cuda()
                        y = y.cuda()
                yhat = self.net(x)
                val_loss += F.cross_entropy(yhat, y)
                yhat_var += yhat.var()

        output_layer_weights = list(self.net.parameters())[-2]
        assert output_layer_weights.size(0) == 10

        state = np.array([
            train_loss.avg,
            val_loss.avg,
            yhat_var.avg,
            output_layer_weights.mean().data,
            output_layer_weights.var().data,
            self.lr
        ], dtype=np.float32)
        reward = -val_loss.avg
        done = self.training_steps > self.num_train_steps
        info = {
            'train_loss': train_loss.avg,
            'val_loss': val_loss.avg,
            'lr': self.lr
        }

        self.info_list.append(info)

        if self.verbose and self.training_steps % (self.num_train_steps//10) == 0:
            print(f'Step {self.training_steps}/{self.num_train_steps}, train loss: {train_loss}, val_loss: {val_loss}, lr: {self.lr}, reward: {reward}')

        return state, reward, done, info


    def reset(self):
        """
        Resets the environment and returns the initial state.
        """
        if self.cuda:
            self.device = self.random_state.randint(0, self.num_devices)
        setproctitle.setproctitle('PPO2-ALRS-v0')
        self.train_generator = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_generator = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.num_train_batches = len(list(self.train_generator))
        self.training_steps = 0
        self.info_list = []
        self.net = self.net_fn()
        if self.cuda:
            with torch.cuda.device(self.device):
                self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.initial_lr)
        self.lr = self.initial_lr
        self.lambda_func = lambda _: self.lr/self.initial_lr
        self.schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_func)

        state, _, _, _ = self.step(action=2)

        return state

    
    def _info_list_to_plot_metrics(self, info_list, label, smooth_kernel_size=7):
        """
        Converts an info list to a tuple of lists ready for rendering.
        """
        assert len(self.info_list) <= len(info_list)
        if len(self.info_list) < len(info_list):
            info_list = info_list[:len(self.info_list)]
        timeline = np.linspace(start=0, stop=self.training_steps, num=len(info_list))
        train_losses = utils.values_from_list_of_dicts(info_list, key='train_loss')
        val_losses = utils.values_from_list_of_dicts(info_list, key='val_loss')
        learning_rates = utils.values_from_list_of_dicts(info_list, key='lr')

        if smooth_kernel_size is not None and len(info_list) >= smooth_kernel_size:
            smoothed_learning_rates = smooth(learning_rates, kernel_size=smooth_kernel_size)
        else:
            smoothed_learning_rates = None

        return timeline, train_losses, val_losses, learning_rates, smoothed_learning_rates, label

    
    def render(self, mode='human', smooth_kernel_size=7):
        """
        Renders current state as a figure.
        """
        assert mode == 'human'
        sns.set(style='whitegrid')
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        plt.ion()
        plt.figure(0, figsize=(16, 4))
        plt.clf()

        experiments = [
            self._info_list_to_plot_metrics(self.info_list, label='Adaptive schedule'),
            self._info_list_to_plot_metrics(utils.load_baseline('initial_lr'), label='Constant (initial LR)'),
        ]

        plt.subplot(1, 3, 1)
        for i, (timeline, train_losses, val_losses, learning_rates, smoothed_learning_rates, label) in enumerate(experiments):
            plt.plot(timeline, train_losses, color=colors[i], label=label)
        plt.xlabel('Training steps')
        plt.ylabel('Log training loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 2)
        for i, (timeline, train_losses, val_losses, learning_rates, smoothed_learning_rates, label) in enumerate(experiments):
            plt.plot(timeline, val_losses, color=colors[i], label=label)
        plt.xlabel('Training steps')
        plt.ylabel('Log validation loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 3)
        for i, (timeline, train_losses, val_losses, learning_rates, smoothed_learning_rates, label) in enumerate(experiments):
            if smoothed_learning_rates is not None:
                smoothed_learning_rates = smooth(learning_rates, kernel_size=smooth_kernel_size)
                plt.plot(timeline, learning_rates, color=colors[i], alpha=0.25)
                plt.plot(timeline, smoothed_learning_rates, color=colors[i], label=label)
            else:
                plt.plot(timeline, learning_rates, color=colors[i], label=label)
        plt.xlabel('Training steps')
        plt.ylabel('Learning rate')
        plt.legend(loc='upper right')

        last_step = len(self.info_list) == (self.num_train_steps//self.update_freq)+1

        plt.tight_layout()
        if last_step:
            plt.savefig('results/experiment.png')
        plt.show()
        plt.draw()
        plt.pause(5 if last_step else 0.001)
