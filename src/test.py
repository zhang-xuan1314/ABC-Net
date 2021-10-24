from utils import MolecularImageDataset,collate_fn
from torch.utils.data import  DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from unet import UNet
from segnet import SegNet
import matplotlib.pyplot as plt
from test_accuracy import compute_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('data/processed_chem.csv')[400000:5000000].copy().reset_index(drop=True)

dataset = MolecularImageDataset(df)

# x,y = dataset[0]
# plt.imshow(x[0,::2,::2])
# plt.figure()
# plt.imshow(y[-1])
# plt.show()

dataloader = DataLoader(dataset,128,collate_fn=collate_fn)
# ,prefetch_factor=10,num_workers=4)

# model = resnet18(26)
model = UNet(in_channels=1 , heads=[1,21,5,1,4,2])
model.load_state_dict(torch.load('unet_model_weights.pkl'))

model = model.to(device)


torch.cuda.empty_cache()
model.eval()

with torch.no_grad():
    results = []
    for i,(imgs, atom_targets,atom_types,atom_charges,
           bond_targets,bond_types,bond_vectors) in enumerate(dataloader):

        imgs = imgs.to(device)
        atom_targets, atom_types, atom_charges = atom_targets.to(device), atom_types.to(device), atom_charges.to(device)
        bond_targets, bond_types, bond_vectors = bond_targets.to(device), bond_types.to(device), bond_vectors.to(device)

        atom_targets_pred, atom_types_pred, atom_charges_pred, bond_targets_pred, bond_types_pred, bond_vectors_pred = model(imgs)

        # atom_targets_acc = compute_accuracy(imgs,atom_targets,atom_targets_pred)
        # atom_types_acc = compute_accuracy(atom_types, atom_types_pred)
        # atom_charges_acc = compute_accuracy(atom_charges, atom_charges_pred)

        bond_targets_acc = compute_accuracy(imgs,bond_targets, bond_targets_pred)
        bond_types_acc = compute_accuracy(bond_types, bond_types_pred)
        bond_vectors_acc = compute_accuracy(bond_vectors, bond_vectors_pred)










