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
		default='fa-mnist',
		help='dataset to use: mnist | cifar10 | fa-mnist'
	)
	parser.add_argument(
		'--num-train',
		type=int,
		default=50000,
		help='number of training samples'
	)
	parser.add_argument(
		'--num-val',
		type=int,
		default=10000,
		help='number of validation samples'
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
		default=2.0,
		help='factor that controls the maximum change of learning per step'
	)
	parser.add_argument(
		'--ppo2-gamma',
		type=float,
		default=0.99,
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
		default=1e-3,
		help='learning rate of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-total-timesteps',
		type=int,
		default=40000,
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
		default='evhcey_10000',
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


def get_random_string(length=6):
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


def load_dataset_and_network(dataset, num_train, num_val):
	"""
	Loads (a subset of) a specified dataset and its associated neural network architecture.
	"""
	if dataset == 'mnist':
		data = load_mnist(num_train=num_train, num_val=num_val)
		net_fn = lambda: networks.MLP(784, 256, 128, 10)

	elif dataset == 'cifar10':
		data = load_cifar10(num_train=num_train, num_val=num_val)
		net_fn = lambda: resnet18(num_classes=10)

	elif dataset == 'fa-mnist':
		data = load_fashion_mnist(num_train=num_train, num_val=num_val)
		net_fn = lambda: LeNet5(num_channels_in=1, num_classes=10, img_dims=(28, 28))

	else:
		raise ValueError(f'Dataset "{dataset}" is not supported.')

	return data, net_fn


def load_cifar10(num_train=50000, num_val=10000):
	"""
	Loads (a subset of) the CIFAR-10 dataset and returns it as a tuple.
	"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[num_train, len(train_dataset)-num_train])
	val_dataset, _ = torch.utils.data.random_split(val_dataset, lengths=[num_val, len(val_dataset)-num_val])

	return train_dataset, val_dataset


def load_fashion_mnist(num_train=50000, num_val=10000):
	"""
	Loads (a subset of) the Fashion-MNIST dataset and returns it as a tuple.
	"""
	transform = transforms.Compose([transforms.ToTensor()])

	train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
	val_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

	train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[num_train, len(train_dataset)-num_train])
	val_dataset, _ = torch.utils.data.random_split(val_dataset, lengths=[num_val, len(val_dataset)-num_val])

	return train_dataset, val_dataset


def load_mnist(filename='data/mnist.npz', num_train=50000, num_val=10000):
	"""
	Loads (a subset of) the grayscale MNIST dataset and returns it as a tuple.
	"""
	data = np.load(filename)

	x_train = data['X_train'][:num_train].astype('float32')
	y_train = data['y_train'][:num_train].astype('int64')

	x_valid = data['X_valid'][:num_val].astype('float32')
	y_valid = data['y_valid'][:num_val].astype('int64')

	train_dataset = Dataset(x_train, y_train)
	val_dataset = Dataset(x_valid, y_valid)

	return train_dataset, val_dataset


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
