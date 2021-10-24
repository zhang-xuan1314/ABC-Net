import pandas as pd 
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit.Chem.MolStandardize

df = pd.read_csv('results/results.csv')
sums = 0
sums2 = 0
sim = 0

for i in range(len(df)):
    smiles_pred = df.loc[i,'smiles_pred']
    smiles = df.loc[i,'Smiles']


    if (type(smiles)!=type('a')) or (type(smiles_pred)!=type('a')):
        continue
    mol1 = Chem.MolFromSmiles(smiles)
    mol2 = Chem.MolFromSmiles(smiles_pred)

    if (mol2 is None) or (mol1 is None):
        continue
        
    smiles = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(smiles)
    smiles_pred = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(smiles_pred)

    if smiles==smiles_pred:
        sums += 1

    mol1 = Chem.MolFromSmiles(smiles)
    mol2 = Chem.MolFromSmiles(smiles_pred)
    
    smiles1 = Chem.MolToSmiles(mol1,canonical=True,isomericSmiles=False)
    smiles2 = Chem.MolToSmiles(mol2,canonical=True,isomericSmiles=False)

    morganfps1 = AllChem.GetMorganFingerprint(mol1, 3)
    morganfps2 = AllChem.GetMorganFingerprint(mol2, 3)
    morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
    if smiles1==smiles2:
        sums2 += 1
    sim += morgan_tani

acc = sums/len(df)
acc2 = sums2/len(df)
sim = sim/len(df)

print(acc)
print(acc2)
print(sim)
