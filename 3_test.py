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
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tr
from models import ClassificationModel
from datasets import RSClassificationDataset



''' Parameters '''
parser = argparse.ArgumentParser(description='Inference and statistical evaluation of a model (adapted or not).')
parser.add_argument('--dataset_target', type=str, default='UCMerced', const=1, nargs='?',
                    help='Target dataset for model evaluation. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--dataset_model', type=str, default='WHU-RS19', const=1, nargs='?',
                    help='Dataset the model to be evaluated was trained on. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--daMethod', type=str, default='', const=1, nargs='?',
                    help='Domain adaptation method, or else empty string if unadapted model. One of {"MMD", "DeepCORAL", "DeepJDOT", ""}.')
parser.add_argument('--backbone', type=str, default='resnet50', const=1, nargs='?',
                    help='Feature extractor backbone to use (default: "resnet50").')
parser.add_argument('--batchSize', type=int, default=32, const=1, nargs='?',
                    help='Inference batch size (default: 32).')
parser.add_argument('--saveResults', type=bool, default=True, const=1, nargs='?',
                    help='Whether to save results (statistics and confusion matrices) to disk or not.')
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
    
    state = torch.load(open(os.path.join(cnnDir, str(startEpoch)+'.pth'), 'rb'))
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
                pred.append(yhat[x])
                target.append(labels[x])
                confmat[yhat[x], labels[x]] += 1

    return pred, target, confmat



if __name__ == '__main__':

    dl_test = loadDataset([2])
    model = loadModel()
    pred, target, confmat = predict(dl_test, model)

    if args.saveResults:
        # save confusion matrix to LaTeX format
        outDir = 'results/{}/{}'.format(
                dataset_target,
                daMethod
            )
        os.makedirs(outDir, exist_ok=True)

        num_class = torch.sum(confmat, dim=0)
        confmat_rel = confmat / num_class.float()

        ua = confmat_rel.diag() / confmat_rel.sum(0).float()        #TODO: check which one is ua and which one is pa
        pa = confmat_rel.diag() / confmat_rel.sum(1).float()

        ua[torch.isnan(ua)] = 0.0
        pa[torch.isnan(pa)] = 0.0

        oa = torch.mean(confmat.diag().float() / num_class.float()).item()
        pa = confmat.diag() / confmat.sum(1).float()
        pa[torch.isnan(pa)] = 0.0
        aa = pa.mean()
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(pred, target)

        print('Dataset: ' + dataset_target)
        print('Model: ' + dataset_model + '/' + daMethod)
        print('\nStatistics:')
        print('OA:\t{:.2f}'.format(oa))
        print('AA:\t{:.2f}'.format(aa))
        print('kappa:\t{:.2f}'.format(kappa))


        def write_confmat(mat, f):
            for idx in range(len(classAssoc_inv)):
                f.write(' & ' + classAssoc_inv[idx])
            f.write(' & UA')
            f.write('\\\\\n')
            f.write('\\hline\\\\\n')
            for x in range(len(classAssoc_inv)):
                f.write(classAssoc_inv[x])
                for y in range(len(classAssoc_inv)):
                    f.write(' & {}'.format(confmat[x,y]))
                f.write(' & {}'.format(ua[x]))
                f.write('\\\\\n')
            f.write('\\hline\\\\\n')
            f.write('PA')
            for idx in range(len(classAssoc_inv)):
                f.write(' & {}'.format(pa[idx]))
            f.write('\\\\\n')
            f.write('\\hline\\\\\n')

        f_abs = open(os.path.join(outDir, 'confusionmat_abs.tex'), 'w')
        f_rel = open(os.path.join(outDir, 'confusionmat_rel.tex'), 'w')
        
        write_confmat(confmat, f_abs)
        write_confmat(confmat_rel, f_rel)

        f_rel.write('\n\nGeneral statistics:\n')
        f_rel.write('OA: {}\n'.format(oa))
        f_rel.write('AA: {}\n'.format(aa))
        f_rel.write('kappa: {}\n'.format(kappa))

        f_abs.close()
        f_rel.close()