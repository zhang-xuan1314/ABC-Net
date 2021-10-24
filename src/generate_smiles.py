import rdkit
from rdkit import Chem

from indigo import IndigoObject, Indigo
from indigo.inchi import IndigoInchi

indigo = Indigo()
indigo_inchi = IndigoInchi(indigo)

def sdf2smiles(atom_list,bond_list,atom_charge_list,bond_type_list,atoms_position_list=None,atom_hs_list=[]):
    """
    :param atom_list:  ['C','N','O','P']
    :param bond_list:  [[1,2],[3,4],[1,3],[3,4]]
    :param atom_charge_list: [1,0,1,-1,0]
    :param bond_type_list: [1,2,3,4]
    :return:
    """
    text = '\n     RDKit\n\n'

    atom_num = str(int(len(atom_list)))
    bond_num = str(int(len(bond_list)))

    atom_num = ' ' * (3 - len(atom_num)) + atom_num
    bond_num = ' ' * (3 - len(bond_num)) + bond_num

    text += '{}{}  0  0  0  0  0  0  0  0999 V2000\n'.format(atom_num,bond_num)

    for i,atom in enumerate(atom_list):
        atom = atom + ' ' * (4-len(atom))
        if atoms_position_list is None:
            text += '    0.0000    0.0000    0.0000 {}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(atom)
        else:
            atoms_position_list[i][0] = atoms_position_list[i][0]/60-1
            atoms_position_list[i][1] = atoms_position_list[i][1] / 60 - 1
            if atoms_position_list[i][0]<0:
                if atoms_position_list[i][1]<0:
                    text += '   {:2.4f}   {:2.4f}    0.0000 {}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(atoms_position_list[i][0],atoms_position_list[i][1],atom)
                else:
                    text += '   {:2.4f}    {:.4f}    0.0000 {}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(
                        atoms_position_list[i][0], atoms_position_list[i][1], atom)
            else:
                if atoms_position_list[i][1]<0:
                    text += '    {:.4f}   {:2.4f}    0.0000 {}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(atoms_position_list[i][0],atoms_position_list[i][1],atom)
                else:
                    text += '    {:.4f}    {:.4f}    0.0000 {}0  0  0  0  0  0  0  0  0  0  0  0\n'.format(
                        atoms_position_list[i][0], atoms_position_list[i][1], atom)

    for i, bond in enumerate(bond_list):
        begin,end = int(bond[0]),int(bond[1])
        begin = str(begin)
        end = str(end)

        begin = ' ' * (3 - len(begin)) + begin
        end = ' ' * (3 - len(end)) + end

        bond_type = int(bond_type_list[i])

        if bond_type <=4:
            bond_type = str(bond_type)
            bond_stereo = '0'

        else:
            if bond_type==5:
                bond_stereo = '1'
            else:
                bond_stereo = '6'

            bond_type = '1'

        bond_type = ' ' * (3 - len(bond_type)) + bond_type
        bond_stereo = ' ' * (3 - len(bond_stereo)) + bond_stereo


        text += '{}{}{}{}\n'.format(begin,end,bond_type,bond_stereo)

    total_charge_num = 0
    charge_line = ''
    for i,charge in enumerate(atom_charge_list):
        if charge != 0:
            total_charge_num += 1

            i,charge = str(i+1),str(charge)

            i = ' '*(4-len(i)) + i
            charge = ' '*(4-len(charge)) + str(int(charge))
            charge_line = charge_line + i + charge

    total_charge_num = str(int(total_charge_num))
    total_charge_num = ' '*(3-len(total_charge_num)) + total_charge_num

    charge_line = 'M  CHG' +total_charge_num+ charge_line + '\n'
    text += charge_line
    
    if len(atom_hs_list)>0:
        text+= 'M  STY  {}'.format(len(atom_hs_list)) + ''.join(['   {} DAT'.format(k+1)   for  k in range(len(atom_hs_list))]) + '\n'
        text+= 'M  SLB  {}'.format(len(atom_hs_list)) + ''.join(['   {}   {}'.format(k+1,k+1)   for  k in range(len(atom_hs_list))]) + '\n'

        for k in range(len(atom_hs_list)):
            text+='M  SAL   {}  1  {}  \n'.format(k+1,atom_hs_list[k])
            text+='M  SDT   {} MRV_IMPLICIT_H    \n'.format(k+1)
            text+='M  SDD   {}     0.0000    0.0000    DA    ALL  1       1    \n'.format(k+1)
            text+='M  SED   {} IMPL_H1\n'.format(k+1)


    text += 'M  END\n$$$$'


    #with open('temp.sdf','w') as f:
    #    f.write(text)
    #    f.close()

    #mols = [mol for mol in Chem.SDMolSupplier('temp.sdf')]    
    #print(text)
    
    mol = Chem.MolFromMolBlock(text)
    if mol is None:
        return None
    smiles = Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
    return smiles
