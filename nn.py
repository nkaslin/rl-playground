import torch.nn as nn


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers.append(nn.Linear(sizes[j], sizes[j+1]))
        layers.append(act())
    return nn.Sequential(*layers)