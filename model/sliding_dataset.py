import math

from torch.utils.data import Dataset


class SlidingDataset(Dataset):
    def __init__(self, dataset, window, step):
        self.dataset = dataset
        self.window = window
        self.step = step

    def __getitem__(self, index):
        batch = []
        for i in range(index*self.step, index*self.step+self.window):
            batch.append(self.dataset[i])
        return batch

    def __len__(self):
        return math.ceil((len(self.dataset) - self.window + 1)/self.step)
