# BitsMNIST.jl
[![Build Status](https://travis-ci.com/nickolasrm/BitsMNIST.jl.svg?branch=main)](https://travis-ci.com/nickolasrm/BitsMNIST.jl)
[![Coverage Status](https://coveralls.io/repos/github/nickolasrm/BitsMNIST.jl/badge.svg?branch=main)](https://coveralls.io/github/nickolasrm/BitsMNIST.jl?branch=main)
[![codecov](https://codecov.io/gh/nickolasrm/BitsMNIST.jl/branch/main/graph/badge.svg?token=CZGSot9qzs)](https://codecov.io/gh/nickolasrm/BitsMNIST.jl)

_Handwritten numbers predicted by bit neural networks_

## Introduction
Bit Neural Networks (BNNs) are a low memory consumption and low-end processors friendly alternative to float32 neural networks (FNNs). It uses a bit per parameter (weights, biases and features), stored in 64-bit floats instead of 32-bit float per parameter. Because of that, BNNs can achieve up to 64 times less memory consumption and up to 32 times speed up when compared to FNNs.

## Usage
### Downloading datasets
Binary Neural networks can accept floats as features. However, treating the dataset by defining explicitly what should become 0 or 1 (bits) is good to make sure of what relevant pixels are gonna be shown. You can download it through these commands:
#### Bits MNIST
Regular MNIST with bits defined by `if pixel > avg_of_pixels_greater_than_zero, then 1, else 0`.
```julia
dataset = BitsMNIST.Datasets.mnist()
Dict{String, Any} with 4 entries:
  "train_y" => [5, 0, 4, 1, 9, …  
  "train_x" => BitVector[[0, 0, 0, 0, 0, ...
  "test_y"  => [7, 2, 1, 0, 4, ...
  "test_x"  => BitVector[[0, 0, 0, 0, 0, ...
```
#### Noisy Bits MNIST
The previous dataset, but added noise in it. `if rand() > 0.3, then pixel = !pixel`
```julia
dataset = BitsMNIST.Datasets.noisymnist()
Dict{String, Any} with 4 entries:
  "train_y" => [-1, -1, -1, -1, -1, ...
  "train_x" => BitVector[[0, 0, 0, 0, 0, ... 
  "test_y"  => [-1, -1, -1, -1, -1, ...
  "test_x"  => BitVector[[0, 0, 0, 0, 0, ...
```
All noisymnist labels have the value defined by the constant `BitsMNIST.Datasets.NOISE_LABEL`

> Once you've downloaded a dataset, it will be stored in a cached folder, so that you'll not need to download it again.

### ZeroOne
Predicting numbers from 0 to 9 can be a CPU intensive task. A simpler case instead can be predicting whether a number is 0 or 1. Let's check it out how to perform this.

First step: Download the dataset
```julia
dset = BitsMNIST.Datasets.mnist()
```

#### Sampling
After downloading the dataset you'll have to take a sample with zeros and ones. Happily, there's a sample function that will extract these examples in a 50/50 proportion.

Second step: Sampling
```julia
sx, sy = BitsMNIST.ZeroOne.sample(set["train_x"], set["train_y"], 0.01)
#0.01 is the fraction of the entire dataset
#Since the dataset has 60000 examples, 0.01*60000 will return 600 examples.
```

#### Defining your model
Through [TinyML](https://github.com/ATISLabs/TinyML.jl/) you can use bit layers to define your bit neural network. Also, you can, and you shall use it with [Flux](https://github.com/FluxML/Flux.jl).

```julia
model = Chain(BitDense(784, 800), BitDense(800, 2, true, σ=sigmoid))
#784 is the number of pixels of an example
#800 is the number of hidden neurons
#2 is the number of classes we want to predict as outputs (0 or 1).
```

> You don't have to import these tools, they are reexported by this project for you to work with.

#### Training Setup
There is a difficult regarding BNNs training. Since the steps of a gradient training are too small to adjust the parameters, an alternative training method should be used. Remember, BNNs parameters can only assume 0 or 1, which means, for example, an adjustment of 0.1 is not really possible to apply. 

##### Gradient
In fact, by modifying gradient to approximate the steps into bits is a possibility [1] [2]. However, this approach is not yet implemented.

##### Reinforcement
As an alternative, reinforcement learning turns out to be a possibility, since the search space is dramatically reduced for these networks.

###### Evaluation function
The first step towards reinforcement learning is to define an evaluation function in order to distinguish when a model is more suited than another. Currently, you can do this by using two functions.

```julia
score_fitness = BitsMNIST.ZeroOne.Reinforcement.generate_score_fitness(sx, sy)
```
This first function increases the score of a model by summing the value of the respective output when predicted correctly. `if predicted_correctly, then score += argmax(model_output)`

```julia
mcc_fitness = BitsMNIST.ZeroOne.Reinforcement.generate_mcc_fitness(sx, sy)
```
This second function increases the score of a model by applying the [Matthews correlation coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC)

###### TrainingSet
Another required step before start training is to configure our genetic algorithm. We do this by creating a [TinyML](https://github.com/ATISLabs/TinyML.jl/)'s Genetic TrainingSet

```julia
tset = Genetic.TrainingSet(
	model, #The model we are gonna train
	model.layers, #The layers we want it to optimize,
	mutationRate=0.05 #Mutation rate reduced to 0.05 for this problem)
```

Other properties can also be configured, but for this example it is enough for what we want to test. Check out these settings at the [TinyML](https://github.com/ATISLabs/TinyML.jl/) page.

#### Training (The hardest part)
After all these steps we can finally train our model.

```julia
Genetic.train!(tset, genNumber=10)
``` 

The most boring part is to wait it finishing...

### Statistics
Checklist: model defined - true, model trained - true. Wait, how can we say our model is trained without a metric? In this case we can call the functions inside the Statistics module in order to test how well our model is performing. Let's use the ZeroOne example to try this out.

#### Error
An easy metric to be visualized is the error. The error is defined as the percentage of error-ed predictions in the total number of examples.
```julia
BitsMNIST.Statistics.error(model, sx, sy)
# This will calculate the error percentage among the sample.
0.05333333333333334
#This means 5.33% of the 600 examples were predicted wrongly.
```

### IO
Let's say you liked your model so much you want to send it to a friend. Well, that is possible through the use of the IO module.

#### Save
```julia
BitsMNIST.IO.save("./mymodel.jld2", model, tset)
```

#### Load
```julia
mymodel = BitsMNIST.IO.load("./mymodel.jld2")
Dict{String, Any} with 2 entries:
  "model" => Chain(BitDense(784, 800), BitDense(800, 2, σ=σ))
  "trainingset" => TrainingSet(popSize=100)
``` 

### References
[1] [Binary Neural Networks: A Survey](https://arxiv.org/pdf/2004.03333.pdf)

[2] [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)

[3] [TinyML](https://github.com/ATISLabs/TinyML.jl/)

[4] [Flux](https://github.com/FluxML/Flux.jl)

[5] [Matthews correlation coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)