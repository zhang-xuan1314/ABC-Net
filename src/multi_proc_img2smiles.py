from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet import UNet
from utils import MolecularImageDataset, collate_fn
plt.switch_backend('agg')
from generate_smiles import sdf2smiles
from utils import atom_vocab, charge_vocab
from rdkit import Chem


def leaky_relu(x):
    x = np.maximum(x, 0.5 * x)
    return x


atom_type_devocab = {j: i for i, j in atom_vocab.items()}
atom_type_devocab[0] = 'C'
atom_charge_devocab = {j: i for i, j in charge_vocab.items()}
bond_type_devocab = {0: 1, 1: 2, 2: 3}
bond_stereo_devocab = {0: 0, 1: 1, 2: 6}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

atom_max_valence = {'<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1, 'P': 5, 'Br': 1,
                    'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3, 'Al': 3, 'Zn': 2,
                    'Ca': 2, 'Ag': 1}


def myfunc(nums, atom_target_img, atom_type_img, atom_charge_img, bond_target_img,
           bond_type_img, bond_stereo_img, bond_rhos_img, bond_omega_img):
    if (atom_target_img.sum()) == 0 or (bond_target_img.sum() == 0):
        print('cannot find key target point')
        results.append(None)
        return None

    bonds_position_list = []
    bonds_property_list = []
    bonds_delta_list = []

    for position in bond_target_img.nonzero(as_tuple=False):
        x, y = position
        x, y = x.cpu().item(), y.cpu().item()
        omega_types = bond_omega_img[:, x, y].cpu().tolist()
        omega_type1 = bond_omega_img[:, x, y].cpu().max().item()
        omega_index1 = bond_omega_img[:, x, y].cpu().argmax().item()
        omega_type2 = -10
        omega_index2 = 0

        for m, omega_type in enumerate(omega_types):
            pre = omega_types[m - 1]
            if m == 29:
                next_ = omega_types[1]
            else:
                next_ = omega_types[m + 1]
            if omega_type >= pre and omega_type > next_ and omega_type > 0:
                if (omega_type <= omega_type1) and omega_type > -1 and (m - omega_index1) > 5:
                    omega_type2 = omega_type
                    omega_index2 = m

        omega = omega_index1 * (np.pi / 30) + np.pi / 60 - np.pi / 2
        rho = bond_rhos_img[omega_index1, x, y].cpu().item()
        delta_x, delta_y = rho * np.cos(omega), rho * np.sin(omega)
        bond_tereo = bond_stereo_img[omega_index1, x, y].cpu().tolist()
        bond_type = bond_type_img[omega_index1, x, y].cpu().item()

        if len(bonds_position_list) > 0:
            temp1 = np.array(bonds_position_list)
            temp2 = np.array([[x, y]])
            if np.sum(np.square(temp1 - temp2), axis=-1).min() < 4:
                continue

        if bond_type != 0:
            bond_tereo = 0

        bonds_position_list.append([x, y])
        bonds_property_list.append([bond_type, bond_tereo])
        bonds_delta_list.append([delta_x, delta_y])

        if omega_type2 > -1:
            omega = omega_index2 * (np.pi / 30) + np.pi / 60 - np.pi / 2
            rho = bond_rhos_img[omega_index2, x, y].cpu().item()
            delta_x, delta_y = rho * np.cos(omega), rho * np.sin(omega)
            bond_tereo = bond_stereo_img[omega_index2, x, y].cpu().tolist()
            bond_type = bond_type_img[omega_index2, x, y].cpu().item()

            if bond_type != 0:
                bond_tereo = 0

            bonds_position_list.append([x, y])
            bonds_property_list.append([bond_type, bond_tereo])
            bonds_delta_list.append([delta_x, delta_y])

    atoms_position_list = []
    atoms_type_list = []
    atoms_charge_list = []

    for position in atom_target_img.nonzero(as_tuple=False):
        x, y = position
        x, y = x.cpu().item(), y.cpu().item()
        atom_type = atom_type_img[x, y].cpu().item()
        atom_charge = atom_charge_img[x, y].cpu().item()
        if len(atoms_position_list) > 0:
            temp1 = np.array(atoms_position_list)
            temp2 = np.array([[x, y]])
            if np.sum(np.square(temp1 - temp2), axis=-1).min() < 4:
                continue

        atoms_position_list.append([x, y])
        atoms_type_list.append(atom_type_devocab[atom_type])
        atoms_charge_list.append(atom_charge_devocab[atom_charge])

    atom_pred_position1 = np.expand_dims(np.array(bonds_position_list) + np.array(bonds_delta_list), 1)
    atom_pred_position2 = np.expand_dims(np.array(bonds_position_list) - np.array(bonds_delta_list), 1)

    atoms_position = np.expand_dims(np.array(atoms_position_list), 0)

    e1 = np.array(bonds_delta_list) / np.sqrt((np.array(bonds_delta_list) ** 2).sum(-1, keepdims=True))
    e2 = np.flip(e1.copy(), 1)

    e2[:, 0] = -e2[:, 0]
    e1 = np.expand_dims(e1, 1)
    e2 = np.expand_dims(e2, 1)

    # distance1 = ((atom_pred_position1 - atoms_position)**2).sum(-1)

    distance1 = np.abs(leaky_relu(((atom_pred_position1 - atoms_position) * e1).sum(-1))) + np.abs(
        (2 * (atom_pred_position1 - atoms_position) * e2).sum(-1))

    # distance2 = ((atom_pred_position2 - atoms_position)**2).sum(-1)

    distance2 = np.abs(leaky_relu(-((atom_pred_position2 - atoms_position) * e1).sum(-1))) + np.abs(
        (2 * (atom_pred_position2 - atoms_position) * e2).sum(-1))

    atom_index1 = distance1.argmin(-1)
    atom_index2 = distance2.argmin(-1)

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

        if (((np.array(atoms_position[0, index1]) + np.array(atoms_position[0, index2])) / 2 - np.array(
                bonds_position_list[i])) ** 2).sum() > 49:
            continue

        if [index1, index2] in bond2atom_index or [index2, index1] in bond2atom_index:
            continue

        bond2atom_index.append([index1, index2])
        bonds_property_list_final.append([bond_type_devocab[bonds_property_list[i][0]], 0])
        bonds_delta_list_final.append(bonds_delta_list[i])
        bonds_position_list_final.append(bonds_position_list[i])

    # average_bond_length = np.mean(np.array(bonds_length))
    atom_showed_list = []

    for atom_index in bond2atom_index:
        atom_showed_list += atom_index

    atom_mask = []

    for i in range(len(atoms_position_list)):
        if i in atom_showed_list:
            atom_mask.append(True)
        else:
            atom_mask.append(False)

    atom_counts = [abs(i) for i in atoms_charge_list]

    for temp, atom_index in enumerate(bond2atom_index):
        x, y = atom_index
        bond_nums = bonds_property_list_final[temp][0]
        atom_counts[x] += bond_nums
        atom_counts[y] += bond_nums

    for serial, atom_count in enumerate(atom_counts):
        atom_type = atoms_type_list[serial]
        if atom_max_valence[atom_type] < atom_count:
            print(serial, atom_count, atom_type)
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
                atom_mask[serial] = False

    corresponding_index = []
    atoms_type_list_final = []
    atoms_charge_list_final = []
    atoms_position_list_final = []

    k = 1
    for i in range(len(atoms_position_list)):
        if atom_mask[i]:
            corresponding_index.append(k)
            atoms_type_list_final.append(atoms_type_list[i])
            atoms_charge_list_final.append(atoms_charge_list[i])
            atoms_position_list_final.append(atoms_position_list[i])
            k += 1
        else:
            corresponding_index.append(k)

    bond2atom_index_final = []

    for atom_index in bond2atom_index:
        x, y = atom_index
        x = corresponding_index[x]
        y = corresponding_index[y]
        bond2atom_index_final.append([x, y])

    try:
        smiles_pred = sdf2smiles(atoms_type_list_final, bond2atom_index_final, atoms_charge_list_final,
                                 bonds_property_list_final, None)

    except:
        return None
    return nums, smiles_pred


df = pd.read_csv('../data/indigo_train_data_test/processed_inchi.csv')[100000:105000].copy().reset_index(drop=True)

# df = pd.read_csv('train_data/processed_inchi.csv')[400000:410000].copy().reset_index(drop=True)
dataset = MolecularImageDataset(df)

# x = dataset[0]
# plt.imshow(x[0])
# plt.show()
# plt.figure()
# plt.imshow(y[-1])
# plt.show()

dataloader = DataLoader(dataset, 64, collate_fn=collate_fn)
# ,prefetch_factor=10,num_workers=4)
# model = resnet18(26)
model = UNet(in_channels=1, heads=[1, 20, 5, 1, 90, 90, 30, 30])
model = nn.DataParallel(model)
model.load_state_dict(torch.load('weights2/unet_model_weights6.pkl'))
model = model.to(device)
torch.cuda.empty_cache()
model.eval()
total_nums = 0
results = [0] * len(dataset)
temp_results = []
p = Pool(32)
total_nums = 0

with torch.no_grad():
    for batch_num, (imgs, atom_targets, atom_types, atom_charges,
                    bond_targets, bond_types, bond_stereos, bond_rhos, bond_omega_types) in enumerate(dataloader):
        imgs = imgs.to(device)
        atom_targets_pred, atom_types_pred, atom_charges_pred, bond_targets_pred, bond_types_pred, bond_stereos_pred, bond_rhos_pred, bond_omega_types_pred = model(
            imgs)
        temp = torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3,
                                              stride=1, padding=1)
        atom_targets_pred = (temp == atom_targets_pred) * (atom_targets_pred > -1).float()
        temp = torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3,
                                              stride=1, padding=1)

        bond_targets_pred = (temp == bond_targets_pred) * (bond_targets_pred > -1).float()
        bond_rhos_pred = torch.abs(bond_rhos_pred)
        bond_types_pred = bond_types_pred.view(-1, 3, 30, 120, 120)
        bond_stereos_pred = bond_stereos_pred.view(-1, 3, 30, 120, 120)

        for j in range(atom_targets_pred.shape[0]):
            img = imgs[j].detach().cpu().numpy()
            atom_target_img = atom_targets_pred[j, 0].detach().cpu()
            atom_type_img = atom_types_pred[j].argmax(0).detach().cpu()
            atom_charge_img = atom_charges_pred[j].argmax(0).detach().cpu()
            bond_target_img = bond_targets_pred[j, 0].detach().cpu()
            bond_type_img = bond_types_pred[j].argmax(0).detach().cpu()
            bond_stereo_img = bond_stereos_pred[j].argmax(0).detach().cpu()
            bond_rhos_img = bond_rhos_pred[j].detach().cpu()
            bond_omega_img = bond_omega_types_pred[j].detach().cpu()

            temp_results.append(p.apply_async(myfunc, args=(
                total_nums, atom_target_img, atom_type_img, atom_charge_img, bond_target_img,
                bond_type_img, bond_stereo_img, bond_rhos_img, bond_omega_img)))
            total_nums += 1

    p.close()
    p.join()

    for temp_result in temp_results:
        nums, smiles_pred = temp_result.get()
        results[nums] = smiles_pred


def smiles2inchi(smiles):
    if smiles is None:
        return None
    # smiles = smiles.split('.')
    # length = 0
    # max_smiles = ''
    # for s in smiles:
    #     if len(s) > length:
    #         max_smiles = s
    # smiles = max_smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    return inchi


def inchi2smiles(inchi):
    if inchi is None:
        return None
    # smiles = smiles.split('.')
    # length = 0
    # max_smiles = ''
    # for s in smiles:
    #     if len(s) > length:
    #         max_smiles = s
    # smiles = max_smiles

    mol = Chem.inchi.MolFromInchi(inchi)
    # mol = Chem.MolToSmiles(mol)
    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    return smiles


df['smiles_pred'] = results
df['InChI'] = df['InChI'].map(inchi2smiles)
dff = df[['InChI', 'smiles_pred']]
dff.to_csv('results/results.csv')
