import os
import torch
import copy
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from urllib.request import urlretrieve

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):

    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)


def download_imagenet(classes, classids):
    path = os.path.abspath("./")
    path = os.path.join(path, "dataset/")
    if not os.path.exists(path):
        print("Downloading Imagenet Dataset ({} Classes):".format(len(classes)))
        os.makedirs(path)

        print("\tDownloading Classes (This will take some time)...")

        for i in range(len(classes)):
         download_class(path, classes[i], classids[i])

        print("Completed Downloading Imagenet Dataset")

def download_class(path, iclass, classid):
    class_path = path + iclass
    list_path = class_path+'/downloads_list.txt'

    if not os.path.exists(class_path):
        os.makedirs(class_path)

    urlretrieve("http://image-net.org/api/text/imagenet.synset.geturls?wnid=" + classid, list_path)

    downloads = []
    with open(list_path, encoding="utf8") as fh:
            downloads = fh.read().split('\n')
            fh.close()

    samples = 0
    for i in range(len(downloads)):
        try:
            file = "{}/{}_{}.jpeg".format(class_path, iclass, i)
            urlretrieve(downloads[i], file)
            if os.stat(file).st_size < 10000:
                os.remove(file)
            else:
                samples += 1
        except BaseException as e:
            #print("{} - {} - {}".format(iclass, downloads[i], e))
            continue

    print("\tCompleted Class {} - {} Samples Downloaded\n".format(iclass,samples))

def split_dataset(dataset, batch_size, val_split=0.15, test_split=0.15):
    train_dataset = dataset
    val_dataset = copy.deepcopy(dataset)
    test_dataset = copy.deepcopy(dataset)

    np.random.seed(0)

    num_samples = len(dataset)
    index_list = list(range(num_samples))
    np.random.shuffle(index_list)
    test_start = num_samples - int(np.floor(test_split * num_samples))
    val_start = test_start - int(np.floor(val_split * num_samples))

    train_idx, val_idx, test_idx = index_list[:val_start], index_list[val_start:test_start], index_list[test_start:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader




