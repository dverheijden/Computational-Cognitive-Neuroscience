
## Problem statement

One of the many unsolved problems of AI is catastrophic forgetting. A paper by Google Deepmind published in 2016 proposes a novel way to tackle this problem[1]. By using their new Neural Network architecture that they called a Progressive Neural Network(PNN). For our project we will be implementing this type of network and we will train it to play atari games, using the openAi gym[2].

Of course, we will use Chainer to implement this network.

We can probably use the implementation we already have from the reinforcement learning problem as a basis for our network. Then we will have to do two things:

1. Implement the PNN.
1. Let the network train on the atari games.
1. Test whether this achieves the desired result.



[1] https://arxiv.org/abs/1606.04671
[2] https://gym.openai.com/envs/