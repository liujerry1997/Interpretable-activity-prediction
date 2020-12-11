import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def convert_smiles_to_numpy(smiles, radius, num_bits):
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features
