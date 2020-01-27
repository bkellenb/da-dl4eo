'''
    Performs domain adaptation on a model;
    one of the following techniques:
    - DeepCORAL
    - DeepJDOT
    - TODO

    2020 Benjamin Kellenberger
'''

import os
import argparse
import glob
from tqdm import trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tr
import ot
from models import ClassificationModel
from datasets import RSClassificationDataset



''' Parameters '''
parser = argparse.ArgumentParser(description='Domain adaptation for a trained base model.')
parser.add_argument('--dataset_source', type=str, default='UCMerced', const=1, nargs='?',
                    help='Source dataset. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--dataset_target', type=str, default='WHU-RS19', const=1, nargs='?',
                    help='Target dataset. One of {"UCMerced", "WHU-RS19"}.')
parser.add_argument('--daMethod', type=str, default='MMD', const=1, nargs='?',
                    help='Domain adaptation method. One of {"MMD", "DeepCORAL", "DeepJDOT"}.')
parser.add_argument('--freezeSource', type=bool, default=True, const=1, nargs='?',
                    help='Whether to freeze the source domain features for adaptation (i.e., predicted by the source model. Default: True).')
parser.add_argument('--trainSource', type=bool, default=True, const=1, nargs='?',
                    help='Whether to add a regular cross-entropy loss on source (default: True).')
parser.add_argument('--backbone', type=str, default='resnet50', const=1, nargs='?',
                    help='Feature extractor backbone to use (default: "resnet50").')
parser.add_argument('--batchSize', type=int, default=32, const=1, nargs='?',
                    help='Training and evaluation batch size (default: 32).')
parser.add_argument('--lr', type=float, default=1e-5, const=1, nargs='?',
                    help='Initial learning rate, reduced by 10 after every 10 epochs (default: 1e-5).')
parser.add_argument('--decay', type=float, default=0.0, const=1, nargs='?',
                    help='Weight decay (default: 0.0).')
parser.add_argument('--numEpochs', type=int, default=100, const=1, nargs='?',
                    help='Number of epochs (default: 100).')
parser.add_argument('--device', type=str, default='cuda:0', const=1, nargs='?',
                    help='Device (default: "cuda:0").')
args = parser.parse_args()




''' Setup '''
dataset_source = args.dataset_source.lower()
dataset_target = args.dataset_target.lower()
daMethod = args.daMethod.lower()

print('Adaptation: {} --> {}, using method {}'.format(
    dataset_source,
    dataset_target,
    daMethod
))

seed=9375322
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

cnnDir_source = 'cnn_states/{}/{}/'.format(dataset_source, dataset_source)
cnnDir_target = 'cnn_states/{}/{}/'.format(dataset_source, daMethod)
os.makedirs(cnnDir_target, exist_ok=True)
print('Saving model in directory "{}"'.format(cnnDir_target))

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



def loadDataset(dataset, setIdx, transform):
    def collate(data):
        imgs = [d[0] for d in data]
        labels = [classAssoc[d[1]] for d in data]
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

    ds = RSClassificationDataset(dataset,
                                    setIdx,
                                    classAssoc,
                                    transform)
    return DataLoader(
        ds,
        batch_size=args.batchSize,
        collate_fn=collate,
        shuffle=True,
        drop_last=(len(ds) > args.batchSize)
        # num_workers=8
    )



def loadModel(cnnDir):
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




def deepCORAL(source, target):
    '''
        Adapted from: https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/deep/DDC_DeepCoral/Coral.py
    '''
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(args.device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(args.device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss



class MMD_loss(nn.Module):
    '''
        Adapted from: https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/deep/DDC_DeepCoral/mmd.py
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.gaussian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss



def deepJDOT(feat_source, feat_target, y_source, pred_target):
    
    sm_target = F.softmax(pred_target, dim=1)

    c0 = torch.cdist(feat_source, feat_target)
    y_source_onehot = torch.FloatTensor(y_source.size(0), pred_target.size(1)).to(args.device).zero_()
    y_source_onehot.scatter_(1, y_source.unsqueeze(1), 1)
    c1 = torch.cdist(y_source_onehot, sm_target)
    c = c0 + 0.1*c1
    gamma = ot.emd(ot.unif(feat_source.size(0)),ot.unif(feat_target.size(0)),c.detach().cpu().numpy(),numItermax=1e6)
    gamma = torch.from_numpy(gamma)
    nnz = torch.nonzero(gamma)

    # DA loss
    loss = F.mse_loss(feat_source[nnz[:,0],:], feat_target[nnz[:,1],:])

    # surrogate target cls loss
    loss += F.cross_entropy(pred_target[nnz[:,1],:], y_source[nnz[:,0]])

    return loss



def doEpoch(dl_source, dl_target, model_source, model_target, epoch, optim=None):

    model_source.eval()
    model_target.train(optim is not None)

    mmd = MMD_loss(kernel_type='linear')

    oa_source_total = 0.0
    oa_target_total = 0.0
    loss_total = 0.0

    def loopDL(dl):
        while True:
            for x in iter(dl): yield x

    num = max(len(dl_source), len(dl_target))
    tBar = trange(num)
    iter_source = loopDL(dl_source)
    iter_target = loopDL(dl_target)
    for t in tBar:
        data_source, labels_source = next(iter_source)
        data_target, labels_target = next(iter_target)

        data_source, labels_source = data_source.to(args.device), labels_source.to(args.device)
        data_target, labels_target = data_target.to(args.device), labels_target.to(args.device)


        # predict source
        if args.freezeSource:
            with torch.no_grad():
                pred_source, fVec_source = model_source(data_source, True)

        elif optim is not None:
            pred_source, fVec_source = model_target(data_source, True)

        else:
            with torch.no_grad():
                pred_source, fVec_source = model_target(data_source, True)


        # predict target
        if optim is not None:
            optim.zero_grad()
            pred_target, fVec_target = model_target(data_target, True)
            pred_sm_target = F.softmax(pred_target, dim=1)

        else:
            with torch.no_grad():
                pred_target, fVec_target = model_target(data_target, True)
                pred_sm_target = F.softmax(pred_target, dim=1)


        # perform DA
        if daMethod == 'mmd':
            loss = mmd(fVec_source, fVec_target)

        elif daMethod == 'deepcoral':
            loss = deepCORAL(fVec_source, fVec_target)

        elif daMethod == 'deepjdot':
            loss = deepJDOT(fVec_source, fVec_target, labels_source, pred_sm_target)
        
        
        if optim is not None:
            # source loss
            if args.trainSource:
                if args.freezeSource:
                    # re-predict source features with target model
                    pred_source, fVec_source = model_target(data_source, True)
                loss += F.cross_entropy(pred_source, labels_source)

            loss.backward()
            optim.step()

        with torch.no_grad():
            pred_sm_source = F.softmax(pred_source, dim=1)
            yhat_source = torch.argmax(pred_sm_source, dim=1)
            yhat_target = torch.argmax(pred_sm_target, dim=1)

        oa_source_total += torch.sum(labels_source==yhat_source).item() / labels_source.size(0)
        oa_target_total += torch.sum(labels_target==yhat_target).item() / labels_target.size(0)
        loss_total += loss.item()

        tBar.set_description_str('[Ep. {}/{} {}] Loss: {:.2f}, OA source: {:.2f}  target: {:.2f}'.format(
            epoch+1, args.numEpochs,
            'train' if optim is not None else 'val',
            loss_total/(t+1),
            100*oa_source_total/(t+1),
            100*oa_target_total/(t+1)
        ))
        tBar.update(1)


    tBar.close()
    loss_total /= num
    oa_source_total /= num
    oa_target_total /= num

    return model_target, loss_total, oa_source_total, oa_target_total



if __name__ == '__main__':

    dl_source = loadDataset(dataset_source, [0], transform_train)
    dl_target = loadDataset(dataset_target, [0], transform_train)
    dl_target_val = loadDataset(dataset_target, [1], transform_val)

    model_source, _, _ = loadModel(cnnDir_source)
    model_target, state, epoch = loadModel(cnnDir_target)
    if epoch == 0:
        print('No target model trained yet; copying from source...')
        model_target, _, _ = loadModel(cnnDir_source)

    optim, scheduler = setupOptimizer(epoch, model_target)

    while epoch < args.numEpochs:

        doEpoch(dl_source, dl_target,
                model_source, model_target,
                epoch, optim)
        scheduler.step()
        doEpoch(dl_source, dl_target_val,
                model_source, model_target,
                epoch, None)
        
        state['model'] = model_target.state_dict()
        epoch += 1
        torch.save(state, open(os.path.join(cnnDir_target, str(epoch)+'.pth'), 'wb'))
