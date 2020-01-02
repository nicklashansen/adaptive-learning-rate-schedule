# Learning an Adaptive Learning Rate Schedule

PyTorch implementation of the "Learning an Adaptive Learning Rate Schedule" paper found here: https://arxiv.org/abs/1909.09712. Work in progress!

## Experimental details

A controller is optimized by PPO2 to generate adaptive learning rate schedules for MLPs trained on MNIST. Both the actor and the critic are MLPs with 2 hidden layers of size 32. The child networks are MLPs with 3 hidden layers and are optimized using Adam with an initial learning rate of 1e-3 and are trained for 1000 steps on the full MNIST training set (50.000 samples) with a batch size of 1000, i.e. 20 epochs. In the near future, experiments on Fashion MNIST (using LeNet-5) and CIFAR-10 (using ResNet) will be released, which is what the original paper uses. I also intend to extent the experiments to not only control learning rate, but also other optimization hyper-parameters such as β1 and β2 for Adam. Robustness of the learned adaptive schedules can likely also be improved by not fixing the starting point (i.e. initial learning rate). Training is distributed across 4 GPUs in my experiments but the code supports a variable number of GPUs.

Experiments are made in both a discrete and continuous setting. In the discrete setting, the controller controls the learning rate by proposing one of the following actions every 10 steps: 1) double the learning rate, 2) halve the learning rate, 3) do nothing. In the continuous setting, the controller instead proposes a real-valued scaling factor [0.5, 2], which allows the controller to modify learning rates with finer granularity. In both settings, Gaussian noise is applied to learning rate updates to facilitate exploration and improve robustness.
Observations for the controller contain information about current training loss, validation loss, variance of predictions, mean and variance of the weights of the output layer as well as the previous learning rate. To make credit assignment easier, the validation loss at each step is used as reward signal rather than the final validation loss. 

## How to run

You can run the experiments by downloading MNIST and extracting it to a `data` directory located in the root of the repository and the running the `main.py` script from the terminal:

```
python main.py --dataset mnist --num-devices 4 --ppo2-lr 1e-4 --ppo2-total-timesteps 1000000
```

Refer to the `parse_args()` function definition in `utils.py` for a full list of command line arguments.

After training, you can evaluate the learned adaptive learning rate schedule and compare it to the two baselines by running

```
python baselines.py
python test.py --test-id [your experiment id]
```


## Results

Below is shown an example of a learned adaptive (discrete) learning rate schedule for an MLP trained on MNIST. A constant learning rate schedule and a step decay + warmup schedule are displayed for comparison. The learned learning rate schedule converges considerably faster than the baseline schedules and achieves a lower validation loss despite its higher training loss, suggesting that the controller has successfully learned to optimize for generalization.

![alrs](https://i.imgur.com/JBrOZUD.png)
