import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class V1_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(V1_CNN, self).__init__()
        self.features_excit = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_inhib = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.regressor = nn.Sequential(
            nn.Linear(60 * 4 * 4 * 6, 48),  # nuber of images * number of output channel * image width * image height
            nn.ReLU(),
            nn.Linear(128, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes),  # 12 classes
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor):  # input is a len 2 list of torch vector of size (batch_size, 60, 8, 12)
        x = x.transpose(1, 0, 2, 3)

        cnn_output_list = []
        for image in x[0]:
            image: torch.Tensor = image.unsqueeze(1)  # image is now (batch_size, 1, 8, 12) I think
            image = self.features_excit(image)
            image = torch.flatten(image, 1)
            cnn_output_list.append(image)

        for image in x[1]:
            image: torch.Tensor = image.unsqueeze(1)  # image is now (batch_size, 1, 8, 12) I think
            image = self.features_inhib(image)
            image = torch.flatten(image, 1)
            cnn_output_list.append(image)
        
        h = torch.concatenate(cnn_output_list, dim=1)
        h = self.regressor(h)
        
        return h


# DATASET CLASS
# load the pickle files, load the csv data, global scaling? Need to find a way to encode the ex an in tuning curve

