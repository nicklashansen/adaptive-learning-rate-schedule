import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
import pickle as pkl
import os
import string
import random
import json
import tensorflow as tf
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common import make_vec_env, schedules, policies
from stable_baselines import PPO2

from smallrl import networks
from torchvision.models.resnet import resnet18
from lenet import LeNet5


def parse_args():
	"""
	Parses command line arguments provided by the user.
	"""
	parser = argparse.ArgumentParser(description='ALRS')
	parser.add_argument(
		'--dataset',
		type=str,
		default='mnist',
		help='dataset to use: mnist | cifar10 | fa-mnist'
	)
	parser.add_argument(
		'--batch-size',
		type=int,
		default=1000,
		help='batch size used for training of trainee networks'
	)
	parser.add_argument(
		'--update-freq',
		type=int,
		default=10,
		help='frequency of learning rate updates in the learned schedule'
	)
	parser.add_argument(
		'--num-train-steps',
		type=int,
		default=1000,
		help='number of actionable training steps of the trainee networks'
	)
	parser.add_argument(
		'--initial-lr',
		type=float,
		default=1e-3,
		help='initial learning rate of trainee networks'
	)
	parser.add_argument(
		'--num-devices',
		type=int,
		default=1,
		help='number of devices used for training of trainee networks: 1 | 2 | 3 | 4'
	)
	parser.add_argument(
		'--discrete',
		type=int,
		default=0,
		help='whether the learned schedule should be discrete or continuous: 0 | 1'
	)
	parser.add_argument(
		'--action-range',
		type=float,
		default=1.1,
		help='factor that controls the maximum change of learning per step'
	)
	parser.add_argument(
		'--ppo2-gamma',
		type=float,
		default=1.0,
		help='discount factor of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-update-freq',
		type=int,
		default=100,
		help='frequency of updates of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-ent-coef',
		type=float,
		default=0.1,
		help='entropy coefficient of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-lr',
		type=float,
		default=1e-2,
		help='learning rate of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-total-timesteps',
		type=int,
		default=50000,
		help='total timesteps of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-nminibatches',
		type=int,
		default=4,
		help='number of minibatches per update of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-noptepochs',
		type=int,
		default=4,
		help='number of epochs when optimizing the surrogate objective function of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-cliprange',
		type=float,
		default=10,
		help='clip range of policy and value function of the PPO2 controller (-1 means no clipping)'
	)
	parser.add_argument(
		'--ppo2-norm-obs',
		type=int,
		default=1,
		help='normalize observations using an EMA: 0 | 1'
	)
	parser.add_argument(
		'--ppo2-norm-reward',
		type=int,
		default=1,
		help='normalize rewards using an EMA: 0 | 1'
	)
	parser.add_argument(
		'--tb-suffix',
		type=str,
		default='none',
		help='optional suffix for the experiment id displayed in tensorboard'
	)
	parser.add_argument(
		'--test-id',
		type=str,
		default='xkus_steps=100k',
		help='experiment id to load and search for schedules when running test.py (mutually exclusive with --test-schedule)'
	)
	parser.add_argument(
		'--test-schedule',
		type=str,
		default='none',
		help='name of the learned schedule to evaluate when running test.py (mutually exclusive with --test-id)'
	)
	args = parser.parse_args()
	assert args.discrete in {0, 1}
	assert args.ppo2_norm_obs in {0, 1}
	assert args.ppo2_norm_reward in {0, 1}
	args.discrete = bool(args.discrete)
	args.ppo2_norm_obs = bool(args.ppo2_norm_obs)
	args.ppo2_norm_reward = bool(args.ppo2_norm_reward)
	args.cuda = torch.cuda.is_available()
	assert args.dataset in {'mnist', 'cifar10', 'fa-mnist'}
	assert args.num_devices in {1, 2, 3, 4}
	
	if args.tb_suffix == 'none':
		args.tb_suffix = None

	return args


def args_to_str(args, separate_lines=True):
	"""
	Pretty-printing of command line arguments.
	"""
	string = str(args)[:-1].replace('Namespace(', '').replace('{', '').replace('\'', '').replace(':', '=').replace(' ', '')
	if separate_lines:
		string = string.replace(',', '\n')
	
	return string


def args_to_file(args, name):
	"""
	Saves command line arguments to a JSON file with the specified name.
	"""
	with open('data/' + name + '.json', 'w', encoding='utf-8') as f:
		json.dump(args, f, default=lambda x: x.__dict__, ensure_ascii=False, indent=4)


def load_args_file_as_dict(name):
	"""
	Loads command line arguments stored as a JSON file with the specified name.
	"""
	with open('data/' + name + '.json', 'r') as f:
		return json.load(f)


def get_random_string(length=4):
	"""
	Generates a random case-invariant string of specified length.
	"""
	return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


class Dataset(torch.utils.data.Dataset):
	"""
	Implements the PyTorch dataset interface.
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


def load_dataset_and_network(dataset,):
	"""
	Loads a specified dataset and its associated neural network architecture.
	"""
	data = load_dataset(dataset)

	dataset_to_network = {
		'mnist': lambda: networks.MLP(784, 256, 128, 10),
		'cifar10': lambda: resnet18(num_classes=10),
		'fa-mnist': lambda: LeNet5(num_channels_in=1, num_classes=10, img_dims=(28, 28))
	}
	assert dataset in dataset_to_network.keys()
	net_fn = dataset_to_network[dataset]

	return data, net_fn


def load_dataset(dataset):
	"""
	Loads a dataset and returns train, val and test partitions.
	"""
	dataset_to_class = {
		'mnist': torchvision.datasets.MNIST,
		'cifar10': 'CIFAR10',
		'fa-mnist': 'FashionMNIST'
	}
	assert dataset in dataset_to_class.keys()
	transform = transforms.Compose([transforms.ToTensor()])

	train_dataset = dataset_to_class[dataset](root='./data', train=True, download=True, transform=transform)
	train_split, val_split = torch.utils.data.random_split(train_dataset, lengths=[len(train_dataset)-10000, 10000])
	test_split = dataset_to_class[dataset](root='./data', train=False, download=True, transform=transform)

	return train_split, val_split, test_split


def make_alrs_env(args):
	"""
	Make a new ALRS environment with parameters specified as command line arguments.
	"""
	from environment import AdaptiveLearningRateOptimizer

	data, net_fn = load_dataset_and_network(dataset=args.dataset)
	train_data, val_data, _ = data[0], data[1], data[2]

	env = make_vec_env(
        env_id=AdaptiveLearningRateOptimizer,
        n_envs=args.num_devices,
        env_kwargs={
            'train_dataset': train_data,
            'val_dataset': val_data,
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

	return env


def make_ppo2_controller(env, args):
	"""
	Make a new PPO2 controller for the given environment using specified command line arguments.
	"""
	lr_schedule = schedules.LinearSchedule(
        schedule_timesteps=args.ppo2_total_timesteps,
        initial_p=args.ppo2_lr,
        final_p=args.ppo2_lr*0.05
    )

	model = PPO2(
        policy=policies.MlpPolicy,
        env=env,
        gamma=args.ppo2_gamma,
        n_steps=args.ppo2_update_freq,
        ent_coef=args.ppo2_ent_coef,
        learning_rate=lr_schedule.value,
        nminibatches=args.ppo2_nminibatches,
        noptepochs=args.ppo2_noptepochs,
        cliprange=args.ppo2_cliprange,
        verbose=0,
        policy_kwargs={
            'act_fun': tf.nn.relu,
            'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}]
        },
        tensorboard_log='data/tensorboard/ppo2_alrs'
    )

	return model
	

def save_baseline(info_list, name):
	"""
	Saves a baseline file locally for future use.
	"""
	path = 'data/baselines/'
	if not os.path.exists(path): os.makedirs(path)
	with open(path+name+'.pkl', 'wb') as f:
		pkl.dump(info_list, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_baseline(name):
	"""
	Loads a stored baseline file for rendering.
	"""
	path = 'data/baselines/'
	try:
		with open(path+name+'.pkl', 'rb') as f:
			return pkl.load(f)
	except:
		return None


class AvgLoss():
	"""
	Utility class that tracks the average loss.
	"""
	def __init__(self):
		self.sum, self.avg, self.n = 0, 0, 0
		self.losses = []

	def __iadd__(self, other):
		try:
			loss = other.cpu().data.numpy()
		except:
			loss = other
		
		if isinstance(other, list):
			self.losses.extend(other)
			self.sum += np.sum(other)
			self.n += len(other)
		else:
			self.losses.append(float(loss))
			self.sum += loss
			self.n += 1

		self.avg = self.sum / self.n

		return self

	def __str__(self):
		return '{0:.4f}'.format(round(self.avg, 4))

	def __len__(self):
		return len(self.losses)


def values_from_list_of_dicts(lst, key):
	"""
	Converts a list of dictionaries to a list of values from the given key.
	"""
	return [d[key] for d in lst] 
