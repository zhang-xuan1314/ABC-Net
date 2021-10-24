from utils_for_test import MolecularImageDataset, collate_fn
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

plt.switch_backend('agg')
from generate_smiles import sdf2smiles
from copy import deepcopy
from utils import atom_vocab, charge_vocab
import rdkit
from rdkit import Chem
import rdkit.Chem.MolStandardize

def leaky_relu(x):
    x = np.maximum(x, 0.5 * x)
    return x

atom_type_devocab = {j: i for i, j in atom_vocab.items()}
atom_type_devocab[0] = 'C'
atom_charge_devocab = {j: i for i, j in charge_vocab.items()}

bond_type_devocab = {0: 1, 1: 2, 2: 3, 3: 4, 4:5,5:6}
bond_stereo_devocab = {0: 0, 1: 1, 2: 6}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

atom_max_valence = {'<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1, 'P': 5, 'Br': 1,
                    'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3, 'Al': 3, 'Zn': 2,
                    'Ca': 2, 'Ag': 1}

df = pd.read_csv('../data/UOB/uob2.csv')
print(len(df))

dataset = MolecularImageDataset(df)

dataloader = DataLoader(dataset, 32, collate_fn=collate_fn)

model = UNet(in_channels=1, heads=[1,14,3,2,1,360,60,60])
model = nn.DataParallel(model)
model.load_state_dict(torch.load('weights/unet_model_weights29.pkl'))

model = model.to(device)

torch.cuda.empty_cache()
model.eval()

total_nums = 0
with torch.no_grad():
    results = []
    for batch_num, imgs in enumerate(dataloader):

        imgs = imgs.to(device)
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
            [bond_omega_types_pred[:, 59:], bond_omega_types_pred, bond_omega_types_pred[:, :1]], dim=1)\
            .permute(0, 2,3,1).reshape(-1, 128 * 128, 62)

        bond_omega_types_pred2 = ((torch.nn.functional.max_pool1d(bond_omega_types_pred2, stride=1, kernel_size=3,
                padding=0).reshape(-1, 128, 128, 60).permute(0, 3, 1,2) == bond_omega_types_pred) * \
                (bond_omega_types_pred > -1)).float()

        # def plot_surf(z):
        #     h, l = z.shape
        #     x, y = np.meshgrid(np.arange(0, h), np.arange(0, l))
        #     ax = plt.subplot(111, projection='3d')
        #     ax.plot_surface(x, y, z)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_zlabel('z')
        #     plt.show()
        #
        # plt.subplot(131)
        # plt.imshow(imgs.cpu().numpy()[39, 0])
        # plt.subplot(132)
        # plt.imshow(atom_targets_pred.cpu().numpy()[39, 0])
        # plt.subplot(133)
        # plt.imshow(bond_targets_pred.cpu().numpy()[39, 0])
        # plt.show()

        # for x, y in zip(*((bond_targets[0]==1).nonzero())):
        #     rho = bond_rhos[:, x, y]
        #     omega = bond_omega_types[:, x, y].argmax() * (np.pi/60) + np.pi/120 - np.pi/2
        #     plt.plot([y - rho* np.sin(omega), y +  rho*np.sin(omega)], [x -  rho*np.cos(omega), x +  rho*np.cos(omega)])

        for j in range(atom_targets_pred.shape[0]):
            smiles = df.loc[total_nums, 'smiles']
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, canonical=True)

            total_nums += 1
            if total_nums % 100 == 0:
                print(total_nums)
            if True:
                img = imgs[j].detach().cpu().numpy()
                atom_target_img = atom_targets_pred[j, 0]
                atom_type_img = atom_types_pred[j].argmax(0)
                atom_charge_img = atom_charges_pred[j].argmax(0)
                atom_hs_img = atom_hs_pred[j].argmax(0)

                bond_target_img = bond_targets_pred[j, 0]
                bond_type_img = bond_types_pred[j].argmax(0)
                bond_rhos_img = bond_rhos_pred[j]
                bond_omega_img = bond_omega_types_pred[j]
                bond_omega_img2 = bond_omega_types_pred2[j]

                if (atom_target_img.sum()) == 0 or (bond_target_img.sum() == 0):
                    print(j, 'cannot find key target point')
                    results.append(None)
                    continue

                bonds_position_list = []
                bonds_property_list = []
                bonds_delta_list = []
                for position in bond_target_img.nonzero(as_tuple=False):
                    x, y = position
                    x, y = x.cpu().item(), y.cpu().item()
                        

                    for omega_index in bond_omega_img2[:, x, y].nonzero(as_tuple=False):

                        omega_index = omega_index.cpu().item()

                        if omega_index <= 28:
                            if bond_omega_img[omega_index, x, y] < bond_omega_img[(omega_index+29):(omega_index+31), x, y].max():
                                continue
                        elif omega_index == 29:
                            if bond_omega_img[omega_index, x, y] < bond_omega_img[(omega_index+29):(omega_index+30), x, y].max() or \
                                    bond_omega_img[omega_index, x, y] < bond_omega_img[0,x,y]:
                                continue

                        elif omega_index == 30:
                            if bond_omega_img[omega_index, x, y] <= bond_omega_img[(omega_index-30):(omega_index-29), x, y].max() or \
                                bond_omega_img[omega_index, x, y] <= bond_omega_img[59, x, y]:
                                continue

                        elif omega_index >= 31:
                            if bond_omega_img[omega_index, x, y] <= bond_omega_img[(omega_index-31):(omega_index-29), x, y].max():
                                continue

                        omega = omega_index * (np.pi / 30) + np.pi / 60 - np.pi / 2

                        rho = bond_rhos_img[omega_index, x, y].cpu().item()

                        delta_x, delta_y = rho * np.cos(omega), rho * np.sin(omega)
                        bond_type = bond_type_img[omega_index, x, y].cpu().item()

                        bond_type = bond_type_img[omega_index, x, y].cpu().item()

                        bonds_position_list.append([x, y])
                        bonds_property_list.append(bond_type)
                        bonds_delta_list.append([delta_x, delta_y])

                atoms_position_list = []
                atoms_type_list = []
                atoms_charge_list = []
                atoms_hs_list = []
                for position in atom_target_img.nonzero(as_tuple=False):
                    x, y = position
                    x, y = x.cpu().item(), y.cpu().item()
                    atom_type = atom_type_img[x, y].cpu().item()
                    atom_charge = atom_charge_img[x, y].cpu().item()
                    atom_h = atom_hs_img[x, y].cpu().item()
                    if len(atoms_position_list) > 0:
                        temp1 = np.array(atoms_position_list)
                        temp2 = np.array([[x, y]])
                        if np.sum(np.square(temp1 - temp2), axis=-1).min() < 4:
                            continue


                    atoms_position_list.append([x, y])
                    atoms_type_list.append(atom_type_devocab[atom_type])
                    atoms_charge_list.append(atom_charge_devocab[atom_charge])
                    atoms_hs_list.append(atom_h)

                atom_pred_position1 = np.expand_dims(np.array(bonds_position_list) + np.array(bonds_delta_list), 1)
                atom_pred_position2 = np.expand_dims(np.array(bonds_position_list) - np.array(bonds_delta_list), 1)
                atoms_position = np.expand_dims(np.array(atoms_position_list), 0)

                e1 = np.array(bonds_delta_list) / np.sqrt((np.array(bonds_delta_list) ** 2).sum(-1, keepdims=True))
                e2 = np.flip(e1.copy(), 1)
                e2[:, 0] = -e2[:, 0]

                e1 = np.expand_dims(e1, 1)
                e2 = np.expand_dims(e2, 1)

                distance1 = np.abs(leaky_relu(((atom_pred_position1 - atoms_position) * e1).sum(-1))) + np.abs(
                    (2 * (atom_pred_position1 - atoms_position) * e2).sum(-1))
                distance2 = np.abs(leaky_relu(-((atom_pred_position2 - atoms_position) * e1).sum(-1))) + np.abs(
                    (2 * (atom_pred_position2 - atoms_position) * e2).sum(-1))

                atom_index1 = distance2.argmin(-1)
                atom_index2 = distance1.argmin(-1)

                bond2atom_index = []
                bonds_property_list_final = []
                bonds_length = []
                bonds_delta_list_final = []
                bonds_position_list_final = []
                for i in range(len(bonds_position_list)):
                    index1 = atom_index1[i]
                    index2 = atom_index2[i]

                    if index1 == index2:
                        continue

                    length = np.sqrt(
                        np.sum((np.array(atoms_position[0, index1]) - np.array(atoms_position[0, index2])) ** 2))
                    bonds_length.append(length)

                    if [index1, index2] in bond2atom_index or [index2, index1] in bond2atom_index:
                        continue

                    bond2atom_index.append([index1, index2])
                    bonds_property_list_final.append(bond_type_devocab[bonds_property_list[i]])
                    bonds_delta_list_final.append(bonds_delta_list[i])
                    bonds_position_list_final.append(bonds_position_list[i])

                atom_showed_list = []
                for atom_index in bond2atom_index:
                    atom_showed_list += atom_index

                atom_mask = []
                for i in range(len(atoms_position_list)):
                    if i in atom_showed_list:
                        atom_mask.append(True)
                    else:
                        atom_mask.append(False)

                atom_counts = [-i for i in atoms_charge_list]
                for temp, atom_index in enumerate(bond2atom_index):
                    x, y = atom_index
                    bond_nums = bonds_property_list_final[temp]

                    if bond_nums == 4 or bond_nums==5 or bond_nums==6:
                        bond_nums = 1

                    atom_counts[x] += bond_nums
                    atom_counts[y] += bond_nums
                for serial, atom_count in enumerate(atom_counts):
                    atom_type = atoms_type_list[serial]
                    if atom_max_valence[atom_type] < atom_count:
                        print(total_nums, serial, atom_count, atom_type)
                        if atom_count == 2:
                            atoms_type_list[serial] = 'O'
                        if atom_count == 3:
                            atoms_type_list[serial] = 'N'
                        if atom_count == 4:
                            atoms_type_list[serial] = 'C'
                        if atom_count == 5:
                            atoms_type_list[serial] = 'P'
                        if atom_count == 6:
                            atoms_type_list[serial] = 'S'
                        if atom_count == 7:
                            atoms_type_list[serial] = 'Cl'

                corresponding_index = []
                atoms_type_list_final = []
                atoms_charge_list_final = []
                atoms_position_list_final = []
                atoms_hs_list_final = []

                k = 1
                for i in range(len(atoms_position_list)):
                    if atom_mask[i]:
                        corresponding_index.append(k)
                        atoms_type_list_final.append(atoms_type_list[i])
                        atoms_charge_list_final.append(atoms_charge_list[i])
                        atoms_position_list_final.append(atoms_position_list[i])
                        atoms_hs_list_final.append(atoms_hs_list[i])
                        k += 1
                    else:
                        corresponding_index.append(k)

                bond2atom_index_final = []

                for atom_index in bond2atom_index:
                    x, y = atom_index
                    x = corresponding_index[x]
                    y = corresponding_index[y]
                    bond2atom_index_final.append([x, y])

                atom_implicit_hs_list = []
                for temp2, atom_index in enumerate(bond2atom_index_final):
                    x, y = atom_index
                    bond_nums = bonds_property_list_final[temp2]
                    if bond_nums == 4:
                        if atoms_type_list_final[x - 1] != 'C':
                            if atoms_hs_list_final[x - 1] != 0:
                                if not x in atom_implicit_hs_list:
                                    atom_implicit_hs_list.append(x)
                        if atoms_type_list_final[y - 1] != 'C':
                            if atoms_hs_list_final[y - 1] != 0:
                                if not y in atom_implicit_hs_list:
                                    atom_implicit_hs_list.append(y)

                smiles_pred = sdf2smiles(atoms_type_list_final, bond2atom_index_final, atoms_charge_list_final,
                                         bonds_property_list_final, deepcopy(atoms_position_list_final),
                                         atom_implicit_hs_list)


                smiles = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(smiles)
                if smiles_pred is not None:
                    smiles_pred = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(smiles_pred)

                results.append(smiles_pred)
                #if smiles!=smiles_pred:
                if False:
                    plt.subplot(131)
                    plt.imshow(img[0])
                    plt.subplot(132)
                    plt.imshow(atom_target_img.detach().cpu().numpy())
                    for m, position in enumerate(atoms_position_list_final):
                        x, y = position
                        position = [y, x]
                        plt.annotate(atoms_type_list_final[m], xy=position, fontsize=6)
                    ax = plt.subplot(133)
                    plt.imshow(bond_target_img.detach().cpu().numpy())
                    for m, position in enumerate(bonds_position_list_final):
                        x, y = position
                        position = [y, x]
                        ax.annotate(bonds_property_list_final[m], xy=position, fontsize=6)
                        x, y = position
                        delta_y, delta_x = bonds_delta_list_final[m]
                        ax.plot([x - delta_x, x + delta_x], [y - delta_y, y + delta_y])
                    a = plt.savefig('results/' + str(total_nums) + '.png', dpi=1000)
                    plt.close()
            else:
                results.append(None)

df['Smiles'] = df['smiles']
df['smiles_pred'] = results
dff = df[['Smiles', 'smiles_pred']]
dff.to_csv('results/results.csv')
