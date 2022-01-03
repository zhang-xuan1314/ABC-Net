from indigo import IndigoObject, Indigo
from indigo.renderer import IndigoRenderer
import cv2
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
"""
aromaticity-model
default:basic
basic:External double bond for aromatic rings are not allowed
generic:External double bond are allowed

render-label-mode:
default:terminal-hetero
all:show all atoms
terminal-hetero:show heteroatoms, terminal atoms, atoms with radical, charge, isotope, explicit valence, and atoms having two adjacent bonds in a line
hetero:the same as terminal-hetero, but without terminal atoms
none:hide all labels, show only bonds

render-implicit-hydrogens-visible
default:True
description
Show implicit hydrogens on visible atoms.

render-stereo-style
default:old
description
Stereocenters rendering mode
old:Only display the “Chiral” sign when appropriate.
ext:Display “abs”, “and”, “or” labels near each stereocenter.
none:Hide all the information about the stereogroups.
"""

IMG_Width = 512

indigo = Indigo()
indigo_render = IndigoRenderer(indigo)
# indigo.setOption("render-margins", 10, 10);
indigo.setOption('render-background-color', "1,1,1")
indigo.setOption('render-implicit-hydrogens-visible', True)
indigo.setOption("aromaticity-model", "generic")
indigo.setOption("dearomatize-verification",False)
indigo.setOption("render-highlight-thickness-enabled",True)


# mol = Chem.MolFromInchi('CCCCC/C=C\C/C=C\CCCCCCCCCC(=O)OCC(COC(=O)CCCCC/C=C\CCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCCC')
bond_stereo_dict = {}
# mol = Chem.MolFromSmiles("CCCCc1ccc(CCCC)cc1")
def smiles2img(smiles,path='temp.png',serial=0):
    size = np.random.randint(320,512)
    indigo.setOption('render-image-width', size)
    indigo.setOption('render-image-height', size)
    indigo.setOption('render-stereo-style', ['none','old'][np.random.randint(0,2)])

    indigo.setOption("render-label-mode", ['all','terminal-hetero','hetero'][np.random.randint(0,3)])

    indigo.setOption("render-bond-line-width", np.random.randint(1,5))

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromInchi(smiles)
        assert mol is not None,"input SMILES is not a valid smiles or inchi string"
    smiles = Chem.MolToSmiles(mol)
    mol = indigo.loadMolecule(smiles)

    if np.random.rand()<0.5:
        mol.dearomatize()
    mol.layout()
    x = []
    y = []
    for i,a in enumerate(mol.iterateAtoms()):
        position = a.xyz()
        x.append(position[0])
        y.append(position[1])
    x = np.array(x)
    y = np.array(y)

    angle = np.random.uniform(0.0,2*np.pi)
    x, y = (x * np.cos(angle) - y * np.sin(angle)), (x * np.sin(angle) + y * np.cos(angle))

    for i, a in enumerate(mol.iterateAtoms()):
        a.setXYZ(x[i], y[i], 0)

    deltas = []
    for bond in mol.iterateBonds():
        begin_atom = bond.source()
        end_atom = bond.destination()
        delta = np.sqrt(
            (x[begin_atom.index()] - x[end_atom.index()]) ** 2 + (y[begin_atom.index()] - y[end_atom.index()]) ** 2)
        deltas.append(delta)

    delta = np.array(deltas).mean()
    bond_length = int(delta*(size)/max(x.max()-x.min(),y.max()-y.min()))
    bond_length = min(max(bond_length*np.random.uniform(2,3),40),200)
    bond_length = int(delta * (size-bond_length) / max(x.max() - x.min(), y.max() - y.min()))
    scale = bond_length / delta

    indigo.setOption("render-bond-length", bond_length)
    indigo_render.renderToFile(mol, path)

    x = x - (x.max() + x.min()) / 2
    y = y - (y.max() + y.min()) / 2
    x = size // 2 + (scale * x).astype('int32')
    y = size // 2 - (scale * y).astype('int32')

    bond_x = []
    bond_y = []
    bond_img_x = []
    bond_img_y = []

    try:
        for bond in mol.iterateBonds():
            bond_type = bond.bondOrder()
            begin_atom = bond.source()
            end_atom = bond.destination()
            if ((bond_type == 1)) and (begin_atom.symbol() == end_atom.symbol()):
                bond_x.append((x[begin_atom.index()] + x[end_atom.index()]) / 2)
                bond_y.append((y[begin_atom.index()] + y[end_atom.index()]) / 2)
                bond.highlight()
                indigo_render.renderToFile(mol, '_temp2.png')
                img_ref = cv2.imread('_temp2.png', flags=1)
                img_ref = (img_ref[:, :, 2] > 230) * (img_ref[:, :, 0] < 230) * (img_ref[:, :, 1] < 230)
                # plt.imshow(img_ref)
                # plt.show()
                red_indices = np.where(img_ref)
                col_indices, row_indices = red_indices
                red_min_x, red_max_x = row_indices.min(), row_indices.max()
                red_min_y, red_max_y = col_indices.min(), col_indices.max()
                red_mean_x = (red_min_x + red_max_x) / 2
                red_mean_y = (red_min_y + red_max_y) / 2
                bond_img_x.append(red_mean_x)
                bond_img_y.append(red_mean_y)
                bond.unhighlight()

        if len(bond_x)<3:
            bond_x = []
            bond_y = []
            bond_img_x = []
            bond_img_y = []
            for bond in mol.iterateBonds():
                bond_type = bond.bondOrder()
                begin_atom = bond.source()
                end_atom = bond.destination()
                if (bond_type == 1 or bond_type==4):
                    bond_x.append((x[begin_atom.index()] + x[end_atom.index()]) / 2)
                    bond_y.append((y[begin_atom.index()] + y[end_atom.index()]) / 2)
                    bond.highlight()
                    indigo_render.renderToFile(mol, '_temp2.png')
                    img_ref = cv2.imread('_temp2.png', flags=1)
                    img_ref = (img_ref[:, :, 2] > 230) * (img_ref[:, :, 0] < 230) * (img_ref[:, :, 1] < 230)
                    # plt.imshow(img_ref)
                    # plt.show()
                    red_indices = np.where(img_ref)
                    col_indices, row_indices = red_indices
                    red_min_x, red_max_x = row_indices.min(), row_indices.max()
                    red_min_y, red_max_y = col_indices.min(), col_indices.max()
                    red_mean_x = (red_min_x + red_max_x) / 2
                    red_mean_y = (red_min_y + red_max_y) / 2
                    bond_img_x.append(red_mean_x)
                    bond_img_y.append(red_mean_y)
                    bond.unhighlight()

        if len(bond_x) < 3:
            return None
    except:
        return None

    if len(bond_x) == 0:
        return None

    bond_x = np.array(bond_x)
    bond_y = np.array(bond_y)
    bond_img_x = np.array(bond_img_x)
    bond_img_y = np.array(bond_img_y)

    x = (x + bond_img_x.mean() - bond_x.mean()).astype('int32')
    y = (y + bond_img_y.mean() - bond_y.mean()).astype('int32')

    img = cv2.imread(path, flags=0)
    img = cv2.resize(img,(512,512))
    cv2.imwrite(path,img)
    x = x*(512/size)
    y = y*(512/size)

    position_x1 = np.array(x).reshape(-1,1)
    position_y1 = np.array(y).reshape(-1,1)
    position_x2 = np.array(x).reshape(1,-1)
    position_y2 = np.array(y).reshape(1,-1)

    if ((x<=0)|(x>=512)|(y<=0)|(y>=512)).any():
        return None

    distance = (position_x1-position_x2)**2 + (position_y1-position_y2)**2 + np.eye(len(x))*150
    if distance.min() <= 100:
        return None

    mol.saveMolfile('_temp2.sdf')
    bond_stereo_dict = {}
    with open('_temp2.sdf') as f:
        mt_data = f.readlines()
        for line in mt_data:
            line = line.strip()
            if len(line) >= 18 and len(line) <= 20 and len(line.split()) == 7:
                stereo_begin = int(line.split()[0]) - 1
                stereo_end = int(line.split()[1]) - 1
                stereo_type = int(line.split()[3])
                if stereo_type == 1 or stereo_type == 6:
                    bond_stereo_dict[(stereo_begin, stereo_end)] = stereo_type
                else:
                    bond_stereo_dict[(stereo_begin, stereo_end)] = 0


    bonds_string = ''

    aromatic_atoms = []
    for i,bond in enumerate(mol.iterateBonds()):

        bond_type = bond.bondOrder()
        begin_atom_idx = bond.source().index()
        end_atom_idx = bond.destination().index()

        if bond_type == 4:
            aromatic_atoms.append(begin_atom_idx)
            aromatic_atoms.append(end_atom_idx)

        if (end_atom_idx,begin_atom_idx) in bond_stereo_dict.keys():
            begin_atom_idx, end_atom_idx  = end_atom_idx,  begin_atom_idx

        bond_stereo_type = bond_stereo_dict[(begin_atom_idx, end_atom_idx)]

        # bond_stereo_type2 = bond.bondStereo()
        # if bond_stereo_type2==5:
        #     bond_stereo_type2 = 1
        # elif bond_stereo_type2 == 6:
        #     bond_stereo_type2 = 6
        # else:
        #     bond_stereo_type2 = 0

        # assert bond_stereo_type == bond_stereo_type2

        y1, x1 = x[begin_atom_idx],y[begin_atom_idx]
        y2, x2 = x[end_atom_idx],y[end_atom_idx]
        xx, yy = (x1 + x2) / 2, (y1 + y2) / 2

        # Indigo.UP — stereo “up” bond:5
        # Indigo.DOWN — stereo “down” bond:6
        # Indigo.EITHER — stereo “either” bond:4
        # Indigo.CIS — “Cis” double bond:7
        # Indigo.TRANS — “Trans” double bond:8
        # zero — not a stereo bond of any kind:0
        if x1 <= x2:
            bond_stereo_direction = 0
        else:
            bond_stereo_direction = 1

        if x1 <= x2:
            delta_x = (x2 - x1) / 2
            delta_y = (y2 - y1) / 2
        else:
            delta_x = (x1 - x2) / 2
            delta_y = (y1 - y2) / 2

        bonds_string += str(bond_type) + ':' + str(int(xx)) + ',' + str(int(yy)) + ',' + str(
            int(delta_x)) + ',' + str(int(delta_y))+ ',' + str(bond_stereo_type) \
                        + ',' + str(bond_stereo_direction) +';'

    atoms_string = ''
    mol.dearomatize()
    for i,atom in enumerate(mol.iterateAtoms()):
        yy, xx = x[i],y[i]
        charge = atom.charge()

        hnums=-1
        if atom.index() in aromatic_atoms:
            if atom.symbol() != 'C':
                hnums = atom.countHydrogens()

        atoms_string += atom.symbol() + ':' + str(int(xx)) + ',' + str(int(yy)) + ',' + str(
            charge) + ',' + str(hnums) +';'


    # img = cv2.imread(path, flags=0)
    # plt.imshow(img)
    # plt.plot(x, y, 'r.')
    # # for bond_string in bonds_string.split(';')[:-1]:
    # #     bond, position = bond_string.split(':')
    # #     x, y, delta_x, delta_y, bond_stereo,_ = position.split(',')
    # #     x, y = int(x), int(y)
    # #     plt.annotate(bond_stereo, xy=(y, x))
    # plt.savefig(path)
    # plt.close()
    return atoms_string,bonds_string

if __name__ == '__main__':
    df = pd.read_csv('filtered.csv').copy().reset_index(drop=True)
    print(len(df))

    df['atoms_string'] = ''
    df['bonds_string'] = ''
    df['path'] = ''
    for i in range(len(df)):
        if i % 1000 == 0:
            print(i)
        m = i%100
        n = m%10
        m = m//10

        smiles = df.loc[i, 'Smiles']
        id = df.loc[i, 'ChEMBL ID']
        path = 'indigo_train_data/images'+ '/' + str(m) + '/' + str(n)
        if not os.path.exists(path):
            os.makedirs(path)

        results = smiles2img(smiles,path+ '/' + id + '.png',i)

        if results:
            atoms_string, bonds_string = results
        else:
            atoms_string, bonds_string = None, None

        df.loc[i, 'atoms_string'] = atoms_string
        df.loc[i, 'bonds_string'] = bonds_string
        df.loc[i, 'path'] = path + '/' + id + '.png'
    df.dropna(axis=0, inplace=True)
    df.to_csv('indigo_train_data/processed_chembl2.csv',index=False)
    print(len(df))
    print(bond_stereo_dict)


