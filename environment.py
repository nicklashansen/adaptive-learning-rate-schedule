import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns

import utils


class AdaptiveLearningRateOptimizer(gym.Env):
    """
    Optimization environment that implements the gym environment interface.
    Can be used to learn an adaptive learning rate schedule.

    Actions:
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

        self.cuda = torch.cuda.is_available()
        assert num_devices == 1 or self.cuda

    
    def _update_lr(self, action, clip=True):
        """
        Updates the current learning rate according to a given action. 
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
            np.log(train_loss.avg),
            np.log(val_loss.avg),
            np.log(yhat_var.avg),
            output_layer_weights.mean().data,
            output_layer_weights.var().data,
            self.lr
        ], dtype=np.float32)
        reward = -np.log(val_loss.avg)
        done = self.training_steps > self.num_train_steps
        info = {'train_loss': train_loss.avg, 'val_loss': val_loss.avg, 'lr': self.lr}

        self.info_list.append(info)

        if self.verbose and self.training_steps % (self.num_train_steps//10) == 0:
            print(f'Step {self.training_steps}/{self.num_train_steps}, train loss: {train_loss}, val_loss: {val_loss}, lr: {self.lr}, reward: {reward}')

        return state, reward, done, info


    def reset(self):
        """
        Resets the environment and returns the initial state.
        """
        if self.cuda:
            self.device = np.random.randint(0, self.num_devices)
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

    
    def render(self, mode='human'):
        """
        Renders current state as a figure.
        """
        assert mode == 'human'
        sns.set(style='whitegrid')
        plt.figure(0, figsize=(16, 4))

        timeline = np.linspace(start=0, stop=self.training_steps, num=len(self.info_list))
        train_losses = utils.values_from_list_of_dicts(self.info_list, key='train_loss')
        val_losses = utils.values_from_list_of_dicts(self.info_list, key='val_loss')
        learning_rates = utils.values_from_list_of_dicts(self.info_list, key='lr')

        plt.subplot(1, 3, 1)
        plt.plot(timeline, train_losses)
        plt.xlabel('Training steps')
        plt.ylabel('Log Training loss')

        plt.subplot(1, 3, 2)
        plt.plot(timeline, val_losses)
        plt.xlabel('Training steps')
        plt.ylabel('Log Validation loss')

        plt.subplot(1, 3, 3)
        plt.plot(timeline, learning_rates)
        plt.xlabel('Training steps')
        plt.ylabel('Log Learning rate')

        plt.tight_layout()
        plt.show()
