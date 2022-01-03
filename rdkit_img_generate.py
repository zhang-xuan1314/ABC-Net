import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO,StringIO
import cv2
import lxml.etree as et
from PIL import Image
import cairosvg

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDrawOptions
import cssutils

bond_type_dict = {rdkit.Chem.rdchem.BondType.SINGLE:0,
            rdkit.Chem.rdchem.BondType.DOUBLE:1,
            rdkit.Chem.rdchem.BondType.TRIPLE:2,
            rdkit.Chem.rdchem.BondType.AROMATIC:3}


np.set_printoptions(edgeitems=30, linewidth=180)

bond_type_dict = {rdkit.Chem.rdchem.BondType.SINGLE:1,
            rdkit.Chem.rdchem.BondType.DOUBLE:2,
            rdkit.Chem.rdchem.BondType.TRIPLE:3,
            rdkit.Chem.rdchem.BondType.AROMATIC:4}

def svg_to_image(svg,convert_to_greyscale=True,add_noise=False):
    svg_str = et.tostring(svg)
    # TODO: would prefer to convert SVG dirrectly to a numpy array.
    png = cairosvg.svg2png(bytestring=svg_str)
    img = np.array(Image.open(BytesIO(png)), dtype=np.float32)
    # Naive greyscale conversion.
    if convert_to_greyscale:
        img = img.mean(axis=-1)

    min_x, min_y, scale = 0,0,1
    # if add_noise:
    #     # Add "salt and pepper" noise.
    #     salt_amount = np.random.uniform(0, 0.2)
    #     salt = np.random.uniform(0, 1, img.shape) < salt_amount
    #     img = np.logical_or(img, salt)
    #     pepper_amount = np.random.uniform(0, 0.02)
    #     pepper = np.random.uniform(0, 1, img.shape) < pepper_amount
    #     img = np.logical_or(1 - img, pepper)
    return img,min_x,min_y,scale


def random_molecule_image(smiles, render_size=512,kekulize=True):
    # Note that the original image is returned as two layers: one for atoms and one for bonds.
    #mol = Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        mol = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mol,clearAromaticFlags=True)
            except:
                mol = Chem.Mol(mol.ToBinary())

    if np.random.rand() < 0.2 and mol.GetNumAtoms()<20:
        mol = Chem.AddHs(mol)
    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)

    sio = StringIO()
    writer = Chem.SDWriter(sio)
    writer.write(mol)
    writer.close()

    mt_data = sio.getvalue()

    bond_stereo_dict = {}
    for line in mt_data.split('\n'):
        line = line.strip()
        if len(line)<=11 and len(line)>=9 and len(line.split())==4:
            stereo_begin = int(line.split()[0])-1
            stereo_end = int(line.split()[1])-1
            stereo_type = int(line.split()[-1])
            if stereo_type==1 or stereo_type==6:
                bond_stereo_dict[(stereo_begin,stereo_end)] = stereo_type
            else:
                bond_stereo_dict[(stereo_begin,stereo_end)] = 0

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(render_size, render_size)
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.padding = np.random.uniform(0.05,0.4)
    options.additionalAtomLabelPadding = np.random.uniform(0.05,0.25)
    options.bondLineWidth = np.random.randint(1,6)
    if np.random.rand()<0.2:
        options.explicitMethyl = True

    options.multipleBondOffset = np.random.uniform(0.1, 0.25)
    options.rotate = np.random.uniform(0, 360)
    # options.fixedScale = np.random.uniform(0.05, 0.07)
    # options.minFontSize = 12
    # options.maxFontSize = options.minFontSize + int(np.round(np.random.uniform(0, 25)))
    # drawer.SetFontSize(2)
    drawer.SetDrawOptions(options)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_str = drawer.GetDrawingText()

    svg = et.fromstring(svg_str.encode('iso-8859-1'))

    # bond_elems = svg.xpath(r'//svg:path[starts-with(@class,"bond-")]', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    # Change the font.
    # font_family = np.random.choice([
    #     'candara',
    #     'calibri light'
    # ])
    if np.random.rand()<0.25:
        atom_elems = svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'})
        for elem in atom_elems:
            style = elem.attrib['style']
            css = cssutils.parseStyle(style)
            css.setProperty('font-weight', 'bold')
            css_str = css.cssText.replace('\n', ' ')
            elem.attrib['style'] = css_str

    img,min_x,min_y,scale = svg_to_image(svg)

    atom_position_x = []
    atom_position_y = []
    atoms_string = ''
    for i in range(mol.GetNumAtoms()):
        y, x = (drawer.GetDrawCoords(i)[0]-min_y)*scale, (drawer.GetDrawCoords(i)[1]-min_x)*scale
        atom_position_x.append(x)
        atom_position_y.append(y)
        charge = mol.GetAtomWithIdx(i).GetFormalCharge()
        atoms_string += mol.GetAtomWithIdx(i).GetSymbol() + ':' + str(int(x)) + ',' + str(int(y)) + ',' + str(
            charge) + ';'

    atom_position_x = np.array(atom_position_x)
    atom_position_y = np.array(atom_position_y)
    position_x1 = np.array(atom_position_x).reshape(-1,1)
    position_y1 = np.array(atom_position_y).reshape(-1,1)
    position_x2 = np.array(atom_position_x).reshape(1,-1)
    position_y2 = np.array(atom_position_y).reshape(1,-1)

    distance = (position_x1 - position_x2) ** 2 + (position_y1 - position_y2) ** 2 + np.eye(len(position_x1)) * 150
    if distance.min() <= 100:
        return None

    bonds_string = ''
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bond_type = bond.GetBondType()
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        if (end_atom_idx,begin_atom_idx) in bond_stereo_dict.keys():
            begin_atom_idx, end_atom_idx  = end_atom_idx,  begin_atom_idx

        bond_stereo_type = bond_stereo_dict[(begin_atom_idx, end_atom_idx)]

        y1, x1 = (drawer.GetDrawCoords(begin_atom_idx)[0]-min_y)*scale, (drawer.GetDrawCoords(begin_atom_idx)[1]-min_x)*scale
        y2, x2 = (drawer.GetDrawCoords(end_atom_idx)[0]-min_y)*scale, (drawer.GetDrawCoords(end_atom_idx)[1]-min_x)*scale
        x, y = (x1 + x2) / 2, (y1 + y2) / 2

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

        bonds_string += str(bond_type_dict[bond_type]) + ':' + str(int(x)) + ',' + str(int(y)) + ',' + str(
            int(delta_x)) + ',' + str(int(delta_y)) + ',' +  str(bond_stereo_type)\
                        + ',' + str(bond_stereo_direction) +';'

    return 1*img,atoms_string,bonds_string

def generate_molecule_graph(path,smiles,id):
    results = random_molecule_image(smiles,512)

    if results:
        img, atoms_string, bonds_string = results

        # from skimage import morphology
        # img = (img < 0.6*255) * 1
        # img = morphology.thin(img, max_iter=np.random.randint(1,4))
        # img = morphology.dilation(img, selem=morphology.disk(1))
        # img = img*255
        # img=255-img
        cv2.imwrite('{}/{}.png'.format(path, id), img=img)
        # plt.imshow(img)
        # for atom_string in atoms_string.split(';')[:-1]:
        #     atom, position = atom_string.split(':')
        #     x, y, charge = position.split(',')
        #     x, y = int(x), int(y)
        #     plt.plot(y, x, 'r.')
        #
        # for bond_string in bonds_string.split(';')[:-1]:
        #     bond, position = bond_string.split(':')
        #     x, y, delta_x, delta_y, bond_stereo,_ = position.split(',')
        #     x, y = int(x), int(y)
        #     plt.annotate(bond_stereo, xy=(y, x))
        # plt.savefig('{}/{}.png'.format(path, id))
        # plt.close()
    else:
        img, atoms_string, bonds_string = None, None, None


    return atoms_string,bonds_string



if __name__ == '__main__':

    df = pd.read_csv('filtered.csv')
    print(len(df))

    df['atoms_string'] = ''
    df['bonds_string'] = ''
    df['path'] = ''
    for i in range(len(df)):
        if i%10000 ==0:
            print(i)

        m = i%100
        n = m%10
        m = m//10

        smiles = df.loc[i,'Smiles']
        id = df.loc[i, 'ChEMBL ID']
        path = 'train_data/images'+ '/' + str(m) + '/' + str(n)
        if not os.path.exists(path):
            os.makedirs(path)
        atoms_string,bonds_string = generate_molecule_graph(path,smiles,id)
        df.loc[i, 'atoms_string'] = atoms_string
        df.loc[i, 'bonds_string'] = bonds_string
        df.loc[i, 'path'] = path + '/' + id + '.png'

    df = df.dropna()
    df.to_csv('train_data/processed_chembl.csv',index=False)



