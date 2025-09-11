from typing import Union, Tuple

import torch
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
            self, input1: torch.Tensor, input2: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
        """
        Simple contrastive loss function with Euclidean distance.
        """
        # Euclidean distance
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        mdist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(mdist, 2)
        loss = torch.sum(loss) / (2.0 * input1.size(0))
        
        return loss

class Flatten(nn.Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)

class SiameseNetwork(nn.Module):

    def __init__(self, contra_loss: bool = False):
        super(SiameseNetwork, self).__init__()

        self.contra_loss = contra_loss

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            Flatten(),
            nn.Linear(512*3*3, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward_once(
            self, x: torch.Tensor) -> torch.Tensor:
        output = self.cnn(x)
        return output

    def forward(
            self, input1: torch.Tensor, input2: torch.Tensor
            ) ->  Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if self.contra_loss:
            return output1, output2
        else:
            output = torch.cat((output1, output2), 1)
            output = self.fc(output)
            return output