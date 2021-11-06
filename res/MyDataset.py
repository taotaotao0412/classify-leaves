import os

from torch.utils.data import Dataset
from configuration import Config
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from sklearn import preprocessing
import numpy as np

def get_img_paths_and_labels(config: Config, df: pd.DataFrame):
    img_names = df['image'].to_list()
    labels = df['label'].to_list()
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels).tolist()
    config.encoder = encoder
    img_paths = list()

    for name in img_names:
        img_paths.append(config.dir + os.sep + name)
    return img_paths, labels

class MyDataset(Dataset):
    def __init__(self, config: Config, image_paths: list, labels: list = None, transform=None):
        self.config = config
        self.image_paths = list()
        self.transform = transform
        self.image_paths = image_paths

        if labels is None:
            self.labels = [x for x in range(len(image_paths))]
        else:
            self.labels = labels

    def __getitem__(self, index):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.config.input_shape),
                transforms.ToTensor()
            ])
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        x, y = image, self.labels[index]
        return x, y

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    config = Config()
    df = pd.read_csv('../dataset/train.csv')
    img_paths, labels = get_img_paths_and_labels(config, df)
    print(labels[:10])
    print(type(labels))