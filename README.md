# Learning an Adaptive Learning Rate Schedule

PyTorch implementation of the "Learning an Adaptive Learning Rate Schedule" paper found here: https://arxiv.org/abs/1909.09712. Work in progress!

## Experimental details

A controller is optimized by PPO2 to generate adaptive learning rate schedules for MLPs trained on MNIST. Both the actor and the critic are MLPs with 2 hidden layers of size 32.
Three distinct child network architectures and datasets are used: 1) an MLP with 3 hidden layers on MNIST, 2) LeNet-5 on Fashion-MNIST and 3) ResNet-18 on CIFAR10. In each of the three settings, child networks are optimized using Adam with an initial learning rate of 1e-3 and are trained for 1000 steps on the full training set (50.000 samples) with a batch size of 1000, i.e. 20 epochs. Learning rate schedules are evaluated based on validation loss. Test loss and test accuracies are in the pipeline. Training is distributed across 4 GPUs in my experiments but the code supports a variable number of GPUs.

Experiments are made in both a discrete and continuous setting. In the discrete setting, the controller controls the learning rate by proposing one of the following actions every 10 steps: 1) increase the learning rate, 2) decrease the learning rate, 3) do nothing. In the continuous setting, the controller instead proposes a real-valued scaling factor, which allows the controller to modify learning rates with finer granularity. In both settings, Gaussian noise is applied to learning rate updates to facilitate exploration and improve robustness.
Observations for the controller contain information about current training loss, validation loss, variance of predictions, mean and variance of the weights of the output layer as well as the previous learning rate. To make credit assignment easier, the validation loss at each step is used as reward signal rather than the final validation loss. Both observations and rewards are normalized by a running mean.

## How to run

You can run the experiments by running the `main.py` script from the terminal:

```
python main.py --dataset cifar10 --discrete 0 --num-devices 4 --ppo2-lr 1e-3 --ppo2-total-timesteps 1000000
```

Refer to the `parse_args()` function definition in `utils.py` for a full list of command line arguments.

After training, you can evaluate the learned adaptive learning rate schedule and compare it to the two baselines by running

```
python baselines.py --dataset cifar10
python test.py --dataset cifar10 --discrete 0 --test-id [your experiment id]
```


## Results

### MLP on MNIST

Below is shown an examples of learned adaptive discrete and continuous learning rate schedules for an MLP trained on MNIST. A constant learning rate schedule and a simple step decay + warmup schedule is displayed for comparison. The learned learning rate schedule converges considerably faster than the baseline schedules and achieves a similar validation loss despite its higher training loss.

In the discrete setting:

![alrs-disc](https://i.imgur.com/JBrOZUD.png)

In the continuous setting:

![alrs-cont](https://i.imgur.com/mksi6Ll.png)


### LeNet-5 on Fashion-MNIST

Next, I learned an adaptive continuous learning rate schedule for LeNet-5 trained on Fashion-MNIST (Fa-MNIST). Baselines similar to those in previous experiments were applied for comparison. Ideally, we would use the learning rate decay function from the official Tensorflow ResNet implementation [available here](https://github.com/tensorflow/models/blob/master/official/r1/resnet/resnet_run_loop.py) as baseline. This experiment shows that the learned schedule outperforms both baselines and shows signs of continued learning beyond the 1000 training step limit that has been set.

![alrs-fa-mnist-cont](https://i.imgur.com/HU8odsa.png)


### ResNet-18 on CIFAR10

Coming soon.
