import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms


def parse_args():
	"""
	Parses command line arguments provided by the user.
	"""
	parser = argparse.ArgumentParser(description='ALRS')
	parser.add_argument(
		'--dataset',
		default='mnist',
		help='dataset to use: mnist | cifar10'
	)
	parser.add_argument(
		'--num-train',
		default=20000,
		help='number of training samples'
	)
	parser.add_argument(
		'--num-val',
		default=4000,
		help='number of validation samples'
	)
	parser.add_argument(
		'--batch-size',
		default=1000,
		help='batch size used for training of trainee networks'
	)
	parser.add_argument(
		'--update-freq',
		default=10,
		help='frequency of learning rate updates in the learned schedule'
	)
	parser.add_argument(
		'--num-train-steps',
		default=1000,
		help='number of actionable training steps of the trainee networks'
	)
	parser.add_argument(
		'--initial-lr',
		default=1e-3,
		help='initial learning rate of trainee networks'
	)
	parser.add_argument(
		'--num-devices',
		default='cpu',
		help='number of devices used for training of trainee networks: cpu | 1 | 2 | 3| 4'
	)
	parser.add_argument(
		'--ppo2-gamma',
		default=0.99,
		help='discount factor of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-update-freq',
		default=100,
		help='frequency of updates of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-lr',
		default=5e-4,
		help='learning rate of the PPO2 controller'
	)
	parser.add_argument(
		'--ppo2-total-timesteps',
		default=100000,
		help='total timesteps of the PPO2 controller'
	)
	args = parser.parse_args()
	
	args.cuda = torch.cuda.is_available()
	if args.num_devices == 'cpu':
		args.num_devices = 1

	assert args.dataset in {'mnist', 'cifar10'}
	assert args.num_devices in {1, 2, 3, 4}

	return args


def args_to_str(args, separate_lines=True):
	"""
	Pretty-printing of command line arguments.
	"""
	string = str(args)[:-1].replace('Namespace(', '').replace(' ', '')
	if separate_lines:
		string = string.replace(',', '\n')
	
	return string


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


def load_cifar(num_train=50000, num_val=2048):
	"""
	Loads a subset of the CIFAR dataset and returns it as a tuple.
	"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=[num_train, len(train_dataset)-num_train])
	val_dataset, _ = torch.utils.data.random_split(val_dataset, lengths=[num_val, len(val_dataset)-num_val])

	return train_dataset, val_dataset


def load_mnist(filename='data/mnist.npz', num_train=4096, num_val=512):
	"""
	Loads a subset of the grayscale MNIST dataset and returns it as a tuple.
	"""
	data = np.load(filename)

	x_train = data['X_train'][:num_train].astype('float32')
	y_train = data['y_train'][:num_train].astype('int64')

	x_valid = data['X_valid'][:num_val].astype('float32')
	y_valid = data['y_valid'][:num_val].astype('int64')

	train_dataset = Dataset(x_train, y_train)
	val_dataset = Dataset(x_valid, y_valid)

	return train_dataset, val_dataset


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
