# https://learn.udacity.com/nanodegrees/nd608/parts/cd13303/lessons/96be0ec2-7a4b-49e8-b8c5-e71a1a927a07/concepts/cd389330-4e38-4b6e-bc45-49a6132f5c52?lesson_tab=lesson
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NumberSumDataset(Dataset):
    def __init__(self, data_range=(1, 10)):
        self.numbers = list(range(data_range[0], data_range[1]))

    def __getitem__(self, index):
        number1 = float(self.numbers[index // len(self.numbers)])
        number2 = float(self.numbers[index % len(self.numbers)])
        return torch.tensor([number1, number2]), torch.tensor([number1 + number2])

    def __len__(self):
        return len(self.numbers) ** 2


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 128)
        self.output_layer = nn.Linear(128, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    dataset = NumberSumDataset(data_range=(0, 100))  ; for i in range(10): print(dataset[i])
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    model = MLP(input_size=2)
    loss_function = nn.MSELoss()

    # Training Loop
    for lr in [0.001, 0.0001, 0.00001 ]:
        optimizer = optim.Adam(model.parameters(), lr=lr) # 0.00001
        for epoch in range(100):
            total_loss = 0.0
            for number_pairs, sums in dataloader:  # Iterate over the batches
                predictions = model(number_pairs)  # Compute the model output
                loss = loss_function(predictions, sums)  # Compute the loss
                loss.backward()  # Perform backpropagation
                optimizer.step()  # Update the parameters
                optimizer.zero_grad()  # Zero the gradients

                total_loss += loss.item()  # Add the loss for all batches

            # Print the loss for this epoch
            print("LR {} | Epoch {}: Sum of Batch Losses = {:.5f}".format(lr, epoch, total_loss))
            # Epoch 0: Sum of Batch Losses = 118.82360

