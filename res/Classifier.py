import torch
from torch import nn
from configuration import Config
from sklearn import preprocessing
from torchvision import models


class Classifier(nn.Module):
    def __init__(self, config: Config, pretrain=True):
        self.config = config
        super().__init__()
        res_net = models.resnet18(pretrained=pretrain)
        self.model = res_net
        fc_features = self.model.fc.in_features
        fc = nn.Sequential(
            nn.Linear(in_features=fc_features, out_features=512),
            nn.Linear(in_features=512, out_features=512),
            nn.Linear(in_features=512, out_features=len(config.encoder.classes_)),
        )
        self.model.fc = fc
        self.model.to(config.device)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    encoder = preprocessing.LabelEncoder()
    encoder.fit(['a','b','c','a'])
    print(len(encoder.classes_))