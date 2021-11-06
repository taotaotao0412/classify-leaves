import os
from sklearn import preprocessing
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import pandas as pd

import engine
from MyDataset import MyDataset, get_img_paths_and_labels
from Classifier import Classifier
from configuration import Config
from engine import train_one_epoch, evaluate


def prepare_data(config: Config, image_paths, labels):
    my_transforms = transforms.Compose([
        transforms.CenterCrop(150),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation((0, 360)),
        transforms.Resize(config.input_shape),
        transforms.ToTensor()
    ])

    full_dataset = MyDataset(config, image_paths, labels)
    train_len = int(config.split_rate * len(full_dataset))
    valid_len = len(full_dataset) - train_len
    train_set, valid_set = torch.utils.data.random_split(full_dataset, [train_len, valid_len])
    train_set.dataset.transforms = my_transforms
    train_dataloader = DataLoader(train_set, config.batch_size, True, num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_set, config.batch_size, False, num_workers=config.num_workers)

    return train_dataloader, valid_dataloader


def main(config):
    df = pd.read_csv('../dataset/train.csv')
    img_paths, labels = get_img_paths_and_labels(config, df)
    train_dataloader, valid_dataloader = prepare_data(config, img_paths, labels)
    classifier = Classifier(config, True)
    optimizer = torch.optim.Adam(classifier.parameters(), config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    if os.path.exists(config.checkpoint):
        print(f'loading check point from: {config.checkpoint}...')
        checkpoint = torch.load(config.checkpoint)
        classifier = checkpoint['model']
        optimizer = checkpoint['optimizer']
        config.start_epoch = checkpoint['epoch']

    print('start training...')

    for epoch in range(config.start_epoch + 1, config.epoch + 1):
        loss, acc = train_one_epoch(classifier, criterion, train_dataloader, optimizer, config.device)
        print(f'[train loss {epoch}/{config.epoch}]: {loss:.5f}')
        print(f'[train acc {epoch}/{config.epoch}]: {acc:.5f}')

        loss, acc = evaluate(classifier, criterion, valid_dataloader, config.device)
        print(f'[evaluate loss {epoch}/{config.epoch}]: {loss:.5f}')
        print(f'[evaluate acc {epoch}/{config.epoch}]: {acc:.5f}')

        print(f'saving checkpoint at {config.checkpoint}...')
        torch.save({
            'model': classifier,
            'optimizer': optimizer,
            'epoch': epoch
        }, config.checkpoint)


if __name__ == '__main__':
    config = Config()
    config.epoch = 26
    main(config)

