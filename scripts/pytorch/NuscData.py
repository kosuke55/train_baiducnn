import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


transform = transforms.Compose([
    transforms.ToTensor()])

# DATA_PATH \
#     = "/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e" \
#     + "/baidu_train_data/all/"

# DATA_PATH \
#     = "/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e" \
#     + "/baidu_train_data/mini/"

# DATA_PATH \
#     = "/media/kosuke/HD-PNFU3/0413/nusc/features_all_640_0421"

DATA_PATH \
    = "/media/kosuke/HD-PNFU3/0413/nusc/mini-0427"

# DATA_PATH \
#     = "/media/kosuke/HD-PNFU3/0413/nusc/features_one_640"

# DATA_PATH \
#     = "/media/kosuke/HD-PNFU3/nusc/feature_instance_one"


class NuscDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir(os.path.join(DATA_PATH, 'in_feature')))

    def __getitem__(self, idx):
        data_name = os.listdir(os.path.join(DATA_PATH, 'in_feature'))[idx]

        in_feature = np.load(os.path.join(DATA_PATH, "in_feature/", data_name))
        in_feature = in_feature.astype(np.float32)

        out_feature = np.load(os.path.join(DATA_PATH, "out_feature/", data_name))

        # use only confidence
        # out_feature = out_feature[..., 0]

        # with class
        # one_hot_class = onehot(out_feature[..., 1].astype(np.int8), 5)
        # out_feature = np.concatenate(
        #     [out_feature[..., 0][..., None], one_hot_class], 2)
        # one_hot_class = onehot(out_feature[..., 4].astype(np.int8), 5)
        one_hot_class = onehot(out_feature[..., 4].astype(np.int8), 6)
        # print(one_hot_class.shape)
        out_feature = np.concatenate(
            [out_feature[..., 0:4], one_hot_class, out_feature[..., 5:]], 2)
        # print(out_feature.shape)
        # import pdb; pdb.set_trace()

        out_feature = torch.FloatTensor(out_feature)
        if self.transform:
            in_feature = self.transform(in_feature)

        return in_feature, out_feature


nusc = NuscDataset(transform)

train_size = int(0.9 * len(nusc))
test_size = len(nusc) - train_size
# train_size = 1
# test_size = 0

train_dataset, test_dataset = random_split(nusc, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=False, num_workers=1)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=1)


if __name__ == '__main__':

    for train_batch in train_dataloader:
        pass
        # print(train_batch)

    for test_batch in test_dataloader:
        pass
        # print(test_batch)
