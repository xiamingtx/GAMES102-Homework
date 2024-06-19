#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:rbfnet.py
# author:xm
# datetime:2024/6/19 17:00
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class RBF(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        # dimensions of input and output features
        self.in_features = in_features
        self.out_features = out_features
        # Initialize the center point, which is a learnable parameter
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        # Initialize the shape parameter beta of the radial basis function, beta is also learnable
        self.beta = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.centers, -1, 1)  # The initial center point value is set to be uniformly distributed
        nn.init.constant_(self.beta, 1)  # The initial value of beta is set to a constant

    def forward(self, x):
        # Extending the dimensions of inputs and centroids for batch processing
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        # Compute the Euclidean distance between the inputs and each center point
        distances = torch.norm(x - centers, dim=-1)
        # Apply Gaussian functions
        return torch.exp(-self.beta * distances ** 2)


class RBFNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super(RBFNet, self).__init__()
        self.rbf = RBF(in_features, hidden_size)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.rbf(x)
        return self.linear(x)


if __name__ == '__main__':
    # Parameters
    input_size = 1
    hidden_size = 10
    output_size = 1

    # Model
    model = RBFNet(input_size, output_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Generate data
    x_train = torch.rand(100, input_size) * 10
    y_train = torch.sin(x_train)

    # Train
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/1000] Loss: {loss.item():.4f}')

    # Visualization
    x_test = torch.linspace(0, 10, 100).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(x_test)

    plt.figure(figsize=(10, 5))
    plt.scatter(x_train.numpy(), y_train.numpy(), color='blue', label='Original Data')
    plt.plot(x_test.numpy(), y_pred.numpy(), color='red', label='Fitted Line')
    plt.xlabel('Input x')
    plt.ylabel('Output y')
    plt.title('RBF Network Fit')
    plt.legend()
    plt.show()
