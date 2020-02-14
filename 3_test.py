'''
    Loads a model and runs it over a specified
    dataset.
    Reports a confusion matrix as well as various
    accuracy measures.

    2020 Benjamin Kellenberger
'''

import os
import argparse
import glob
import numpy as np
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tr
from models import ClassificationModel
from datasets import RSClassificationDataset



''' Parameters '''
parser = argparse.ArgumentParser(description='Inference and statistical evaluation of a model (adapted or not).')
parser.add_argument('--dataset_target', type=str, default='WHU-RS19', const=1, nargs='?',
                    help='Target dataset for model evaluation. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--dataset_model', type=str, default='UCMerced', const=1, nargs='?',
                    help='Dataset the model to be evaluated was trained on. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--daMethod', type=str, default='', const=1, nargs='?',
                    help='Domain adaptation method, or else empty string if unadapted model. One of {"MMD", "DeepCORAL", "DeepJDOT", ""}.')
parser.add_argument('--backbone', type=str, default='resnet50', const=1, nargs='?',
                    help='Feature extractor backbone to use (default: "resnet50").')
parser.add_argument('--batchSize', type=int, default=32, const=1, nargs='?',
                    help='Inference batch size (default: 32).')
parser.add_argument('--visualize', type=bool, default=True, const=1, nargs='?',
                    help='Whether to visualize results (confusion matrices; default: True).')
parser.add_argument('--saveResults', type=bool, default=False, const=1, nargs='?',
                    help='Whether to save results (statistics and confusion matrices) to disk or not (default: False).')
parser.add_argument('--device', type=str, default='cuda:0', const=1, nargs='?',
                    help='Device (default: "cuda:0").')
args = parser.parse_args()



''' Setup '''
dataset_model = args.dataset_model.lower()
dataset_target = args.dataset_target.lower()
daMethod = (args.daMethod.lower() if len(args.daMethod) else dataset_model)

print('Testing model {}/{} on dataset {}'.format(
    dataset_model,
    daMethod,
    dataset_target
))

seed=9375322
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

cnnDir_baseline = 'cnn_states/{}/{}/'.format(dataset_model, dataset_model)
cnnDir = 'cnn_states/{}/{}/'.format(dataset_model, daMethod)

from classAssoc import classAssoc, classAssoc_inv

transform = tr.Compose([
    tr.Resize((128,128)),
    tr.ToTensor(),
    tr.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])



def loadDataset(setIdx):
    def collate(data):
        imgs = [d[0] for d in data]
        labels = [classAssoc[d[1]] for d in data]
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

    return DataLoader(
        RSClassificationDataset(dataset_target,
            setIdx,
            classAssoc,
            transform),
        batch_size=args.batchSize,
        collate_fn=collate,
        shuffle=False
    )



def loadModel(baseline=False):
    if baseline:
        modelDir = cnnDir_baseline
    else:
        modelDir = cnnDir

    model = ClassificationModel(len(classAssoc_inv),
                                args.backbone,
                                pretrained=True,
                                convertToInstanceNorm=False)
    startEpoch = 0
    modelStates = glob.glob(os.path.join(modelDir, '*.pth'))
    for sf in modelStates:
        epoch, _ = os.path.splitext(sf.replace(modelDir, ''))
        startEpoch = max(startEpoch, int(epoch))
    
    state = torch.load(open(os.path.join(modelDir, str(startEpoch)+'.pth'), 'rb'), map_location=lambda storage, loc: storage)
    model.load_state_dict(state['model'])
    print('Loaded model epoch {}.'.format(startEpoch))

    model.to(args.device)
    return model



def predict(dataloader, model):

    model.eval()

    pred = []
    target = []
    confmat = torch.zeros([len(classAssoc_inv), len(classAssoc_inv)], dtype=torch.long)       # pred, target

    for idx, (data, labels) in enumerate(tqdm(dataloader)):

        data = data.to(args.device)

        with torch.no_grad():
            logits = model(data)
            pred_sm = F.softmax(logits, dim=1)
            yhat = torch.argmax(pred_sm, dim=1).cpu()

            for x in range(labels.size(0)):
                pred.append(yhat[x].item())
                target.append(labels[x].item())
                confmat[yhat[x].item(), labels[x].item()] += 1

    return pred, target, confmat



if __name__ == '__main__':

    num_class = len(classAssoc_inv)
    dl_test = loadDataset([2])

    if cnnDir != cnnDir_baseline:
        # do baseline first
        print('Calculating baseline performance for comparison...')
        model = loadModel(baseline=True)
        _, _, confmat_baseline = predict(dl_test, model)
        confmat_baseline_rel = confmat_baseline.float() / confmat_baseline.sum(0)
    else:
        confmat_baseline = None
        confmat_baseline_rel = None
    
    model = loadModel(baseline=False)
    pred, target, confmat = predict(dl_test, model)
    confmat_rel = confmat.float() / confmat.sum(0)


    ua = confmat_rel.diag() / confmat_rel.sum(0).float()
    pa = confmat_rel.diag() / confmat_rel.sum(1).float()

    ua[torch.isnan(ua)] = 0.0
    pa[torch.isnan(pa)] = 0.0

    oa = torch.mean(confmat.diag().float() / confmat.sum(0))
    pa = confmat.diag() / confmat.sum(1).float()
    pa[torch.isnan(pa)] = 0.0
    aa = pa.mean()
    kappa = cohen_kappa_score(pred, target)

    print('Dataset: ' + dataset_target)
    print('Model: ' + dataset_model + '/' + daMethod)
    print('\nStatistics:')
    print('OA:\t{:.2f}'.format(oa))
    print('AA:\t{:.2f}'.format(aa))
    print('kappa:\t{:.2f}'.format(kappa))


    if args.visualize:
        figsize = (4.5, 4)
        classlist = []
        for key in classAssoc_inv.keys():
            classlist.append(classAssoc_inv[key])

        plt.figure(num=1, figsize=figsize)
        mbp = plt.imshow(confmat_rel, cmap='inferno')
        plt.clim(0, 1)
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.xticks(range(confmat_rel.size(0)), classlist)
        plt.yticks(range(confmat_rel.size(0)), classlist)
        plt.box(on=None)
        plt.draw()
        plt.pause(0.001)

        plt.figure(num=2, figsize=figsize)
        ax = plt.gca()
        plt.colorbar(mbp)
        ax.remove()
        plt.draw()
        plt.pause(0.001)

        
        if confmat_baseline is not None:
            # plot baseline
            plt.figure(num=3, figsize=figsize)
            plt.imshow(confmat_baseline_rel, cmap='inferno')
            plt.xlabel('Ground Truth')
            plt.ylabel('Prediction')
            plt.xticks(range(confmat_rel.size(0)), classlist)
            plt.yticks(range(confmat_rel.size(0)), classlist)
            plt.box(on=None)
            plt.draw()
            plt.pause(0.001)


            # plot differences to baseline
            confmat_diff = (confmat_rel - confmat_baseline_rel).numpy()
            colormap = ListedColormap(np.concatenate((np.flip(cm.get_cmap(plt.get_cmap('Reds'))(np.linspace(0.0, 1.0, 100))[:,:3], 0),
                                        np.array([[1.0, 1.0, 1.0]]),
                                        cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, 100))[:,:3]),0))

            plt.figure(num=4, figsize=figsize)
            mbp = plt.imshow(confmat_diff, cmap=colormap)
            plt.clim(-0.6, 0.6)
            plt.xlabel('Ground Truth')
            plt.ylabel('Prediction')
            plt.xticks(range(confmat_rel.size(0)), classlist)
            plt.yticks(range(confmat_rel.size(0)), classlist)
            plt.box(on=None)
            plt.draw()
            plt.pause(0.001)

            plt.figure(num=6, figsize=figsize)
            ax = plt.gca()
            plt.colorbar(mbp)
            ax.remove()
            plt.draw()
            plt.pause(0.001)

        plt.show()


    if args.saveResults:
        # save statistics
        outDir = 'results/{}/{}'.format(
                dataset_target,
                daMethod
            )
        os.makedirs(outDir, exist_ok=True)

        f_abs = open(os.path.join(outDir, 'stats_abs.tex'), 'w')
        f_rel = open(os.path.join(outDir, 'stats_rel.tex'), 'w')

        f_rel.write('\n\nGeneral statistics:\n')
        f_rel.write('OA: {}\n'.format(oa))
        f_rel.write('AA: {}\n'.format(aa))
        f_rel.write('kappa: {}\n'.format(kappa))

        f_abs.close()
        f_rel.close()