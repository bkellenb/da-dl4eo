import os
from torch.utils.data import Dataset
from PIL import Image


class RSClassificationDataset(Dataset):
    def __init__(self, datasetName, split, classes_subset=None, transform=None):
        super(RSClassificationDataset, self).__init__()
        if datasetName.lower() == 'ucmerced':
            self.dataRoot = 'datasets/UCMerced_LandUse'
        else:
            self.dataRoot = 'datasets/WHU-RS19'
        self.split = split
        if isinstance(self.split, int):
            self.split = [self.split]
        self.classes_subset = classes_subset
        self.transform = transform
        
        self.data = []
        with open(os.path.join(self.dataRoot, 'split.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            tokens = line.strip().split(' ')
            setIdx = int(tokens[2])
            if not setIdx in self.split:
                continue
            label = tokens[1]
            if self.classes_subset is not None and label not in self.classes_subset:
                continue
            self.data.append((tokens[0], label))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, x):
        imgPath, label = self.data[x % len(self.data)]
        img = Image.open(os.path.join(self.dataRoot, imgPath)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label





def createSplitFile(dataRoot, dataset='UCMerced', split=(0.7,0.1,0.2)):

    if not dataRoot.endswith('/'):
        dataRoot += '/'
    dataset = dataset.lower()
    data = {}

    # find images
    import glob
    import re
    pattern = re.compile('.*(_|-)')

    if dataset == 'ucmerced':
        imgs = glob.glob(os.path.join(dataRoot, '**/*.tif'), recursive=True)
        for img in imgs:
            parent, name = os.path.split(img)
            _, label = os.path.split(parent)
            label = label.lower()
            if not label in data:
                data[label] = []
            data[label].append(img.replace(dataRoot, ''))

    else:
        imgs = glob.glob(os.path.join(dataRoot, '**/*.jpg'), recursive=True)
        for img in imgs:
            _, name = os.path.split(img)
            name, _ = os.path.splitext(name)
            label = name[0:pattern.match(name).span()[1]-1]
            label = label.lower()
            if not label in data:
                data[label] = []
            data[label].append(img.replace(dataRoot, ''))

    
    # create split
    import numpy as np
    with open(os.path.join(dataRoot, 'split.txt'), 'w') as f:
        f.write('imagePath label split\n')
        for labelclass in data.keys():
            imgs = data[labelclass]

            num_train = np.ceil(len(imgs) * split[0]).astype(np.int)
            num_val = np.ceil(len(imgs) * split[1]).astype(np.int)
            num_test = len(imgs) - num_train - num_val
            num = [num_train, num_val, num_test]

            order = np.random.permutation(len(imgs))

            for s in range(3):
                for n in range(num[s]):
                    f.write('{} {} {}\n'.format(
                        imgs[order[n]],
                        labelclass,
                        s)
                    )





if __name__ == '__main__':
    createSplitFile('datasets/UCMerced_LandUse', 'UCMerced', split=(0.6,0.1,0.3))
    createSplitFile('datasets/WHU-RS19', 'WHU-RS19', split=(0.6,0.1,0.3))
