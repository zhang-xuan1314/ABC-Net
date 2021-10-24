from utils import MolecularImageDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unet import UNet
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from generate_smiles import sdf2smiles
from copy import deepcopy
from utils import atom_vocab, charge_vocab
import rdkit
from rdkit import Chem
from meter import AverageMeter


def leaky_relu(x):
    x = np.maximum(x, 0.5 * x)
    return x


atom_type_devocab = {j: i for i, j in atom_vocab.items()}
atom_type_devocab[0] = 'C'
atom_charge_devocab = {j: i for i, j in charge_vocab.items()}

bond_type_devocab = {0: 1, 1: 2, 2: 3, 3: 4,4:5,5:6}

# tp tn fp fn
atom_detection_metrics = np.zeros((14,4))
atom_type_metrics = np.zeros((14,4))
atom_charge_metrics = np.zeros((3,4))

bond_detection_metrics = np.zeros((6,4))
bond_type_metrics = np.zeros((6,4))




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

atom_max_valence = {'<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1, 'P': 5, 'Br': 1,
                    'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3, 'Al': 3, 'Zn': 2,
                    'Ca': 2, 'Ag': 1}

df1 = pd.read_csv('../data/train_data/processed_chembl.csv')[90000:99000].copy().reset_index(drop=True)
df2 = pd.read_csv('../data/indigo_train_data/processed_chembl.csv')[90000:99000].copy().reset_index(drop=True)

df = pd.concat([df1,df2]).reset_index(drop=True)


dataset = MolecularImageDataset(df)

dataloader = DataLoader(dataset, 64, collate_fn=collate_fn)

model = UNet(in_channels=1, heads=[1, 14, 3, 2, 1, 360, 60, 60])
model = nn.DataParallel(model)
model.load_state_dict(torch.load('weights8/unet_model_weights13.pkl'))

model = model.to(device)

torch.cuda.empty_cache()
model.eval()

train_atom_targets_precision = AverageMeter()
train_atom_targets_recall = AverageMeter()
train_atom_targets_precision3 = AverageMeter()
train_atom_targets_recall3 = AverageMeter()


train_atom_types_acc = AverageMeter()
train_atom_charges_acc = AverageMeter()
train_atom_hs_acc = AverageMeter()

train_bond_targets_precision = AverageMeter()
train_bond_targets_recall = AverageMeter()
train_bond_targets_precision3 = AverageMeter()
train_bond_targets_recall3 = AverageMeter()

train_bond_types_acc = AverageMeter()
train_bond_rhos_mae = AverageMeter()

train_bond_omega_precision = AverageMeter()
train_bond_omega_recall = AverageMeter()
train_bond_omega_precision3 = AverageMeter()
train_bond_omega_recall3 = AverageMeter()

total_nums = 0
with torch.no_grad():
    results = []
    for batch_num, (imgs, atom_targets, atom_types, atom_charges, atom_hs,
                    bond_targets, bond_types, bond_rhos, bond_omega_types) in enumerate(dataloader):

        imgs = imgs.to(device)
        atom_targets, atom_types, atom_charges,atom_hs = atom_targets.to(device), atom_types.to(device), atom_charges.to(device),atom_hs.to(device)
        bond_targets, bond_types,bond_rhos,bond_omega_types = bond_targets.to(device), bond_types.to(device),bond_rhos.to(device),bond_omega_types.to(device)

        atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred, bond_targets_pred, \
        bond_types_pred, bond_rhos_pred, bond_omega_types_pred = model(
            imgs)

        temp = torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3,
                                              stride=1, padding=1)
        atom_targets_pred = (temp == atom_targets_pred) * (atom_targets_pred > -1).float()

        temp = torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3,
                                              stride=1, padding=1)
        bond_targets_pred = (temp == bond_targets_pred) * (bond_targets_pred > -1).float()

        bond_rhos_pred = torch.abs(bond_rhos_pred)

        bond_types_pred = bond_types_pred.view(-1, 6, 60, 128, 128)

        bond_omega_types_pred2 = torch.cat(
            [bond_omega_types_pred[:, 59:], bond_omega_types_pred, bond_omega_types_pred[:, :1]], dim=1) \
            .permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)

        bond_omega_types_pred = ((torch.nn.functional.max_pool1d(bond_omega_types_pred2, stride=1, kernel_size=3,
                                                                 padding=0).reshape(-1, 128, 128, 60).permute(0, 3, 1,
                                                                                                              2) == bond_omega_types_pred) * \
                                 (bond_omega_types_pred > -1)).float()
        atom_targets = (atom_targets==1).float()
        bond_targets = (bond_targets==1).float()

        for i in range(14):
            atom_detection_metrics[i, 0] += ((atom_types.argmax(1,keepdims=True)==i)*(atom_targets_pred
                    * torch.nn.functional.max_pool2d(atom_targets, kernel_size=3, padding=1,stride=1))).sum().detach().cpu().numpy()

            atom_detection_metrics[i, 2] += ((atom_types.argmax(1, keepdims=True) == i) * (atom_targets_pred
                    * (torch.nn.functional.max_pool2d(atom_targets, kernel_size=3, padding=1, stride=1)==0))).sum().detach().cpu().numpy()

            atom_detection_metrics[i, 3] += ((atom_types.argmax(1, keepdims=True) == i) * ((torch.nn.functional.max_pool2d(
                atom_targets_pred,kernel_size=3,padding=1,stride=1)==0)* atom_targets)).sum().detach().cpu().numpy()

            atom_type_metrics[i,0] += (torch.sum(atom_types==1, dim=1) * \
                ((atom_types.argmax(1)==i) * (atom_types_pred.argmax(1)==i))).sum().detach().cpu().numpy()

            atom_type_metrics[i,1] += (torch.sum(atom_types==1, dim=1) * \
                ((atom_types.argmax(1)!=i) * (atom_types_pred.argmax(1)!=i))).sum().detach().cpu().numpy()

            atom_type_metrics[i,2] += (torch.sum(atom_types==1, dim=1) * \
                ((atom_types.argmax(1)!=i) * (atom_types_pred.argmax(1)==i))).sum().detach().cpu().numpy()

            atom_type_metrics[i,3] += (torch.sum(atom_types==1, dim=1) * \
                ((atom_types.argmax(1)==i) * (atom_types_pred.argmax(1)!=i))).sum().detach().cpu().numpy()


        for i in range(3):
            atom_charge_metrics[i,0] += (torch.sum(atom_charges==1, dim=1) * \
                ((atom_charges.argmax(1)==i) * (atom_charges_pred.argmax(1)==i))).sum().detach().cpu().numpy()

            atom_charge_metrics[i, 1] += (torch.sum(atom_charges == 1, dim=1) * \
                ((atom_charges.argmax(1) != i) * (atom_charges_pred.argmax(1) != i))).sum().detach().cpu().numpy()

            atom_charge_metrics[i, 2] += (torch.sum(atom_charges == 1, dim=1) * \
                ((atom_charges.argmax(1) != i) * (atom_charges_pred.argmax(1) == i))).sum().detach().cpu().numpy()

            atom_charge_metrics[i, 3] += (torch.sum(atom_charges == 1, dim=1) * \
                ((atom_charges.argmax(1) == i) * (atom_charges_pred.argmax(1) != i))).sum().detach().cpu().numpy()


        for i in range(6):
            bond_detection_metrics[i, 0] += ((bond_types.sum(2).argmax(1, keepdims=True) == i) * (bond_targets_pred
            * torch.nn.functional.max_pool2d(bond_targets, kernel_size=3, padding=1, stride=1))).sum().detach().cpu().numpy()

            bond_detection_metrics[i, 2] += ((bond_types.sum(2).argmax(1, keepdims=True) == i) * (bond_targets_pred
            * (torch.nn.functional.max_pool2d(bond_targets,kernel_size=3,padding=1,stride=1) == 0))).sum().detach().cpu().numpy()

            bond_detection_metrics[i, 3] += ((bond_types.sum(2).argmax(1, keepdims=True) == i) * ((torch.nn.functional.max_pool2d(
            bond_targets_pred, kernel_size=3, padding=1,stride=1) == 0) * bond_targets)).sum().detach().cpu().numpy()

        for i in range(6):
            bond_type_metrics[i, 0] += (torch.sum(bond_types == 1, dim=1) * \
                ((bond_types.argmax(1) == i) * (bond_types_pred.argmax(1) == i))).sum().detach().cpu().numpy()

            bond_type_metrics[i, 1] += (torch.sum(bond_types == 1, dim=1) * \
                ((bond_types.argmax(1) != i) * (bond_types_pred.argmax(1) != i))).sum().detach().cpu().numpy()

            bond_type_metrics[i, 2] += (torch.sum(bond_types == 1, dim=1) * \
                ((bond_types.argmax(1) != i) * (bond_types_pred.argmax(1) == i))).sum().detach().cpu().numpy()

            bond_type_metrics[i, 3] += (torch.sum(bond_types == 1, dim=1) * \
                ((bond_types.argmax(1) == i) * (bond_types_pred.argmax(1) != i))).sum().detach().cpu().numpy()

        train_atom_targets_precision.update(
            ((atom_targets_pred * atom_targets).sum() / atom_targets_pred.sum()).cpu().detach().numpy(),
            atom_targets_pred.sum().cpu().detach().numpy())
        train_atom_targets_precision3.update(
            ((atom_targets_pred * torch.nn.functional.max_pool2d(atom_targets, kernel_size=3, padding=1,
            stride=1)).sum() / atom_targets_pred.sum()).cpu().detach().numpy(),
            atom_targets_pred.sum().cpu().detach().numpy())

        train_atom_targets_recall.update(
            ((atom_targets * atom_targets_pred).sum() / atom_targets.sum()).cpu().detach().numpy(),
            atom_targets.sum().cpu().detach().numpy())
        train_atom_targets_recall3.update(
            ((atom_targets * torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3, padding=1,
               stride=1)).sum() / atom_targets.sum()).cpu().detach().numpy(),
            atom_targets.sum().cpu().detach().numpy())

        train_atom_types_acc.update((torch.sum(torch.sum(atom_types, dim=1) * ((atom_types.argmax(1) ==
            atom_types_pred.argmax(1)).float())) / torch.sum(
            atom_types)).cpu().detach().numpy(), torch.sum(atom_types).cpu().detach().numpy())

        train_atom_charges_acc.update((torch.sum(torch.sum(atom_charges, dim=1) * ((atom_charges.argmax(1) ==
                                                                                    atom_charges_pred.argmax(
                                                                                        1)).float())) / torch.sum(
            atom_charges)).cpu().detach().numpy(), torch.sum(atom_charges).cpu().detach().numpy())

        train_atom_hs_acc.update((torch.sum(torch.sum(atom_hs, dim=1) * ((atom_hs.argmax(1) ==
            atom_hs_pred.argmax(1)).float())) / (0.01 + torch.sum(atom_hs))).cpu().detach().numpy(),
                                 (0.01 + torch.sum(atom_hs)).cpu().detach().numpy())

        train_bond_targets_precision.update(
            ((bond_targets_pred * bond_targets).sum() / bond_targets_pred.sum()).cpu().detach().numpy(),
            bond_targets_pred.sum().cpu().detach().numpy())
        train_bond_targets_precision3.update(
            ((bond_targets_pred * torch.nn.functional.max_pool2d(bond_targets, kernel_size=3, padding=1,
            stride=1)).sum() / bond_targets_pred.sum()).cpu().detach().numpy(),
            bond_targets_pred.sum().cpu().detach().numpy())

        train_bond_targets_recall.update(
            ((bond_targets * bond_targets_pred).sum() / bond_targets.sum()).cpu().detach().numpy(),
            bond_targets.sum().cpu().detach().numpy())
        train_bond_targets_recall3.update(
            ((bond_targets * torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3, padding=1,
            stride=1)).sum() / bond_targets.sum()).cpu().detach().numpy(),
            bond_targets.sum().cpu().detach().numpy())

        train_bond_types_acc.update((torch.sum(torch.sum(bond_types, dim=1) * (
            (bond_types.argmax(1) == bond_types_pred.argmax(1)).float())) / torch.sum(
            bond_types)).cpu().detach().numpy(), torch.sum(bond_types).cpu().detach().numpy())

        train_bond_rhos_mae.update(
            (torch.sum(torch.abs(bond_rhos_pred - bond_rhos) * torch.sum(bond_types, dim=1)) / \
             torch.sum(bond_types)).cpu().detach().numpy(), torch.sum(bond_types).cpu().detach().numpy())

        temp = torch.cat([bond_omega_types_pred[:, 59:], bond_omega_types_pred, bond_omega_types_pred[:, :1]],
                         dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
        temp = ((torch.nn.functional.max_pool1d(temp, stride=1, kernel_size=3, padding=0).reshape(-1, 128, 128,
                60).permute(0, 3,1,2) == bond_omega_types_pred) * \
                (bond_omega_types_pred > 0.25)).float() * bond_targets

        bond_omega_types = (bond_omega_types == 1)
        train_bond_omega_precision.update((((bond_omega_types) * temp).sum() / (temp).sum()).cpu().detach().numpy(),
                                          temp.sum().cpu().detach().numpy())

        temp2 = torch.cat([temp[:, 59:], temp, temp[:, :1]],
                          dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
        temp2 = (torch.nn.functional.max_pool1d(temp2.float(), stride=1, kernel_size=3, padding=0).reshape(-1, 128, 128,
            60).permute(0,3,1,2)).float()

        train_bond_omega_recall3.update(
            (((bond_omega_types) * temp2).sum() / (bond_omega_types).sum()).cpu().detach().numpy(),
            bond_omega_types.sum().cpu().detach().numpy())

        train_bond_omega_recall.update(
            (((bond_omega_types) * temp).sum() / (bond_omega_types).sum()).cpu().detach().numpy(),
            bond_omega_types.sum().cpu().detach().numpy())

        temp3 = torch.cat([bond_omega_types[:, 59:], bond_omega_types, bond_omega_types[:, :1]],
                          dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
        temp3 = (torch.nn.functional.max_pool1d(temp3.float(), stride=1, kernel_size=3, padding=0).reshape(-1, 128, 128,
            60).permute(0,3,1,2)).float()
        train_bond_omega_precision3.update((((temp3) * temp).sum() / (temp).sum()).cpu().detach().numpy(),
                                           temp.sum().cpu().detach().numpy())

print(atom_detection_metrics)
print(atom_type_metrics)
print(atom_charge_metrics)
print(bond_detection_metrics)
print(bond_type_metrics)

print('sum___')
print('atom_detection:',atom_detection_metrics[:,0]+atom_detection_metrics[:,3])
print('atom_type:',atom_type_metrics[:,0]+atom_type_metrics[:,3])
print('atom_charge:',atom_charge_metrics[:,0]+atom_charge_metrics[:,3])
print('bond_detection:',bond_detection_metrics[:,0]+bond_detection_metrics[:,3])
print('bond_type:',bond_type_metrics[:,0]+bond_type_metrics[:,3])


atom_detection_precision = atom_detection_metrics[:,0]/(atom_detection_metrics[:,0]+atom_detection_metrics[:,2]+1e-4)
atom_detection_recall = atom_detection_metrics[:,0]/(atom_detection_metrics[:,0]+atom_detection_metrics[:,3]+1e-4)

atom_type_precision = atom_type_metrics[:,0]/(atom_type_metrics[:,0]+atom_type_metrics[:,2]+1e-4)
atom_type_recall = atom_type_metrics[:,0]/(atom_type_metrics[:,0]+atom_type_metrics[:,3]+1e-4)

atom_charge_precision = atom_charge_metrics[:,0]/(atom_charge_metrics[:,0]+atom_charge_metrics[:,2]+1e-4)
atom_charge_recall = atom_charge_metrics[:,0]/(atom_charge_metrics[:,0]+atom_charge_metrics[:,3]+1e-4)

bond_detection_precision = bond_detection_metrics[:,0]/(bond_detection_metrics[:,0]+bond_detection_metrics[:,2]+1e-4)
bond_detection_recall = bond_detection_metrics[:,0]/(bond_detection_metrics[:,0]+bond_detection_metrics[:,3]+1e-4)

bond_type_precision = bond_type_metrics[:,0]/(bond_type_metrics[:,0]+bond_type_metrics[:,2]+1e-4)
bond_type_recall = bond_type_metrics[:,0]/(bond_type_metrics[:,0]+bond_type_metrics[:,3]+1e-4)



print('special metrics')
print('atom_detection_precision:',atom_detection_precision)
print('atom_detection_recall:',atom_detection_recall)
print('atom_type_precision:',atom_type_precision)
print('atom_type_recall:',atom_type_recall)
print('atom_charge_precision:',atom_charge_precision)
print('atom_charge_recall:',atom_charge_recall)

print('bond_detection_precision:',bond_detection_precision)
print('bond_detection_recall:',bond_detection_recall)
print('bond_type_precision:',bond_type_precision)
print('bond_type_recall:',bond_type_recall)



print('common metrics')

print('atom_target_precision:', train_atom_targets_precision.avg)
print('atom_target_recall:', train_atom_targets_recall.avg)
print('atom_target_precision3:', train_atom_targets_precision3.avg)
print('atom_target_recall3:', train_atom_targets_recall3.avg)

print('atom_types_acc:', train_atom_types_acc.avg)
print('atom_charges_acc:', train_atom_charges_acc.avg)
print('atom_hs_acc:', train_atom_hs_acc.avg)

print('bond_target_precision:', train_bond_targets_precision.avg)
print('bond_target_recall:', train_bond_targets_recall.avg)
print('bond_target_precision3:', train_bond_targets_precision3.avg)
print('bond_target_recall3:', train_bond_targets_recall3.avg)

print('bond_types_acc:', train_bond_types_acc.avg)
print('bond_rhos_mae:', train_bond_rhos_mae.avg)

print('bond_omega_precision:', train_bond_omega_precision.avg)
print('bond_omega_recall:', train_bond_omega_recall.avg)
print('bond_omega_precision3:', train_bond_omega_precision3.avg)
print('bond_omega_recall3:', train_bond_omega_recall3.avg)
