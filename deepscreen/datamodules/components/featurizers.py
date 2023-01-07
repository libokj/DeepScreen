import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from torch_geometric.utils import from_smiles

SMILES_CHARSET = {'?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                  '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                  'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                  'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                  'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y'}


FASTA_CHARSET = {'?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z'}


def smiles_to_onehot(smiles: str, max_sequence_length: int = 100, in_channels: int = len(SMILES_CHARSET)):
    smiles_character_index = {character: index for index, character in enumerate(SMILES_CHARSET)}
    onehot = np.zeros((max_sequence_length, len(smiles_character_index)))

    for index, character in enumerate(smiles[:max_sequence_length]):
        onehot[index, smiles_character_index.get(character, 0)] = 1

    return onehot.transpose()


def fasta_to_onehot(amino_acid_sequence: str, max_sequence_length: int = 1000, in_channels: int = len(FASTA_CHARSET)):
    fasta_character_index = {character: index for index, character in enumerate(FASTA_CHARSET)}
    onehot = np.zeros((max_sequence_length, len(fasta_character_index)))

    for index, character in enumerate(amino_acid_sequence[:max_sequence_length]):
        onehot[index, fasta_character_index.get(character, 0)] = 1

    return onehot.transpose()


def smiles_to_erg(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        features = np.array(GetErGFingerprint(mol))
    except:
        print(f'RDKit could not find this SMILES: {smiles} convert to all 0 features')
        features = np.zeros((315,))
    return features


def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print(f'RDKit could not find this SMILES: {smiles} convert to all 0 features')
        features = np.zeros((n_bits,))
    return features


def smiles_to_rdkit2d(smiles):
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(smiles)[1:])
        nans = np.isnan(features)
        features[nans] = 0
    except:
        print(f'descriptastorus could not find this SMILES: {smiles} convert to all 0 features')
        features = np.zeros((200,))
    return np.array(features)


def smiles_to_daylight(smiles):
    try:
        NumFinger = 2048
        mol = Chem.MolFromSmiles(smiles)
        bv = FingerprintMols.FingerprintMol(mol)
        temp = tuple(bv.GetOnBits())
        features = np.zeros((NumFinger,))
        features[np.array(temp)] = 1
    except:
        print(f'RDKit could not find this SMILES: {smiles} convert to all 0 features')
        features = np.zeros((2048,))
    return np.array(features)


def smiles_to_graph(smiles):
    return from_smiles(smiles)
