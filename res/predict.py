import pandas as pd
import torch

from configuration import Config
from MyDataset import MyDataset, get_img_paths_and_labels
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == '__main__':
    config = Config()
    df = pd.read_csv('../dataset/train.csv')
    get_img_paths_and_labels(config ,df)
    df = pd.read_csv('../dataset/test.csv')
    images_names = df['image'].tolist()
    image_path = list()
    for name in images_names:
        image_path.append('../dataset/' + name)
    dataset = MyDataset(config, image_path)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=False)
    models = torch.load(config.checkpoint)
    model = models['model']
    model.eval()
    res = list()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels = batch
            logistic = model(images.to(config.device))
            logistic = logistic.argmax(dim=-1).cpu()
            res.extend(logistic.tolist())
    print(res[:10])
    labels = config.encoder.inverse_transform(res)
    with open('../res.txt', 'w') as file:
        file.write('image,label\n')
        for i in range(len(res)):
            file.write(f'{images_names[i]},{labels[i]}\n')
