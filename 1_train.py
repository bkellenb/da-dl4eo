'''
    Trains a base model on a Remote Sensing
    image classification dataset.

    2020 Benjamin Kellenberger
'''

import os
import argparse
import glob
from tqdm import trange
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tr
from models import ClassificationModel
from datasets import RSClassificationDataset


''' Parameters '''
parser = argparse.ArgumentParser(description='Base model training.')
parser.add_argument('--dataset', type=str, default='WHU-RS19', const=1, nargs='?',
                    help='Source dataset. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--backbone', type=str, default='resnet50', const=1, nargs='?',
                    help='Feature extractor backbone to use (default: "resnet50").')
parser.add_argument('--batchSize', type=int, default=32, const=1, nargs='?',
                    help='Training and evaluation batch size (default: 32).')
parser.add_argument('--lr', type=float, default=1e-4, const=1, nargs='?',
                    help='Initial learning rate, reduced by 10 after every 10 epochs (default: 1e-4).')
parser.add_argument('--decay', type=float, default=0.0, const=1, nargs='?',
                    help='Weight decay (default: 0.0).')
parser.add_argument('--numEpochs', type=int, default=100, const=1, nargs='?',
                    help='Number of epochs (default: 100).')
parser.add_argument('--device', type=str, default='cuda:0', const=1, nargs='?',
                    help='Device (default: "cuda:0").')
args = parser.parse_args()





''' Setup '''
dataset = args.dataset.lower()
seed=9375322
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

cnnDir = 'cnn_states/{}/{}/'.format(dataset, dataset)
os.makedirs(cnnDir, exist_ok=True)

from classAssoc import classAssoc, classAssoc_inv

transform_train = tr.Compose([
    tr.Resize((128,128)),
    tr.RandomHorizontalFlip(p=0.5),
    tr.RandomVerticalFlip(p=0.5),
    tr.ColorJitter(0.1, 0.1, 0.1, 0.01),
    tr.ToTensor(),
    tr.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

transform_val = tr.Compose([
    tr.Resize((128,128)),
    tr.ToTensor(),
    tr.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])



def loadDataset(setIdx, transform):
    def collate(data):
        imgs = [d[0] for d in data]
        labels = [classAssoc[d[1]] for d in data]
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

    return DataLoader(
        RSClassificationDataset(dataset,
            setIdx,
            classAssoc,
            transform),
        batch_size=args.batchSize,
        collate_fn=collate,
        shuffle=True
        # num_workers=8
    )



def loadModel():
    model = ClassificationModel(len(classAssoc_inv),
                                args.backbone,
                                pretrained=True,
                                convertToInstanceNorm=False)
    startEpoch = 0
    modelStates = glob.glob(os.path.join(cnnDir, '*.pth'))
    for sf in modelStates:
        epoch, _ = os.path.splitext(sf.replace(cnnDir, ''))
        startEpoch = max(startEpoch, int(epoch))
    
    if startEpoch > 0:
        state = torch.load(open(os.path.join(cnnDir, str(startEpoch)+'.pth'), 'rb'), map_location=lambda storage, loc: storage)
        model.load_state_dict(state['model'])
        print('Loaded model epoch {}.'.format(startEpoch))
    else:
        state = {
            'model': None,
            'loss_train': [],
            'loss_val': [],
            'oa_train': [],
            'oa_val': []
        }
        print('Initialized new model.')
    model.to(args.device)
    return model, state, startEpoch



def setupOptimizer(epoch, model):
    step_size = 10
    gamma = 0.1
    currentLR = args.lr
    if epoch > 1:
        currentLR *= gamma ** (epoch//step_size)

    params = model.parameters()
    optim = Adam(params, lr=currentLR, weight_decay=args.decay)
    for group in optim.param_groups:
            group.setdefault('initial_lr', currentLR)
    scheduler = StepLR(optim, step_size=10, gamma=0.1, last_epoch=epoch-2)

    return optim, scheduler



def doEpoch(dataloader, model, epoch, optim=None):

    model.train(optim is not None)

    oa_total = 0.0
    loss_total = 0.0

    tBar = trange(len(dataloader))
    for idx, (data, target) in enumerate(dataloader):

        data, target = data.to(args.device), target.to(args.device)

        if optim is None:
            with torch.no_grad():
                logits = model(data)
                if logits.dim()==1:
                    logits = logits.unsqueeze(0)
                loss = F.cross_entropy(logits, target)
        else:
            optim.zero_grad()
            logits = model(data)
            if logits.dim()==1:
                logits = logits.unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optim.step()
    
        with torch.no_grad():
            pred_sm = F.softmax(logits, dim=1)
            yhat = torch.argmax(pred_sm, dim=1)
            oa = torch.sum(target==yhat).item() / target.size(0)
        
        loss_total += loss.item()
        oa_total += oa

        tBar.set_description_str('[Ep. {}/{} {}] Loss: {:.2f}, OA: {:.2f}'.format(
            epoch+1, args.numEpochs,
            'train' if optim is not None else 'val',
            loss_total/(idx+1),
            100*oa_total/(idx+1)
        ))
        tBar.update(1)
    
    tBar.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)

    return model, loss_total, oa_total



if __name__ == '__main__':
    
    dl_train = loadDataset([0], transform_train)
    dl_val = loadDataset([1], transform_val)

    model, state, epoch = loadModel()
    optim, scheduler = setupOptimizer(epoch, model)

    while epoch < args.numEpochs:

        model, loss_train, oa_train = doEpoch(dl_train, model, epoch, optim)
        scheduler.step()
        _, loss_val, oa_val = doEpoch(dl_val, model, epoch, None)

        state['model'] = model.state_dict()
        state['loss_train'].append(loss_train)
        state['loss_val'].append(loss_val)
        state['oa_train'].append(oa_train)
        state['oa_val'].append(oa_val)
        epoch += 1
        torch.save(state, open(os.path.join(cnnDir, str(epoch)+'.pth'), 'wb'))