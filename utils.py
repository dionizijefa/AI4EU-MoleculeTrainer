from standardiser import standardise
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from torch_geometric.data import DataLoader, Data
from torch import Tensor, cat
from rdkit import RDConfig, Chem
from rdkit.Chem import HybridizationType, ChemicalFeatures
from pathlib import Path
import numpy as np
from torch.utils.data import WeightedRandomSampler


fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))


def task_type(data, target_col):
    if not ((data.iloc[0][target_col] == 0) or (data.iloc[0][target_col] == 1)):
        problem = 'regression'
    else:
        if data[target_col].value_counts(normalize=True)[1] > 0.3:
            problem = 'auc'
        else:
            problem = 'ap'

    return problem

def standardise_dataset(dataset, smiles_col):
    """ Runs standardizer for molecular prediction tasks"""
    standardized = []
    for drug in dataset[smiles_col]:
        try:
            standardized.append(standardise.run(drug))
        except:
            standardized.append(0)
    dataset['standardized_smiles'] = standardized
    dataset = dataset.loc[dataset['standardized_smiles'] != 0]  # drop cols where standardiser failed
    return dataset


def cross_val(data, target_col, problem, batch_size, seed):
    """Don't split rest of the splits on scaffolds"""

    if problem != 'regression':
        cv_splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
    else:
        cv_splitter = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

    loaders = []
    if problem != 'regression':
        for k, (train_index, test_index) in enumerate(
                cv_splitter.split(data, data[target_col])
        ):

            test = data.iloc[test_index]
            test_data_list = []
            for index, row in test.iterrows():
                test_data_list.append(smiles2graph(row, target_col))
            test_loader = DataLoader(test_data_list, num_workers=0, batch_size=batch_size)

            train_set = data.iloc[train_index]

            train, val = train_test_split(
                train_set,
                test_size=0.15,
                stratify=train_set[target_col],
                shuffle=True,
                random_state=seed
            )

            train_data_list = []
            for index, row in train.iterrows():
                train_data_list.append(smiles2graph(row, target_col))

            # if we are doing classification use weighted sampling for the minority class
            minority = train[target_col].value_counts()[1]
            majority = train[target_col].value_counts()[0]
            class_sample_count = [majority, minority]
            weights = 1 / Tensor(class_sample_count)
            samples_weights = weights[train[target_col].values]
            sampler = WeightedRandomSampler(samples_weights,
                                            num_samples=len(samples_weights),
                                            replacement=True)
            train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                      sampler=sampler, drop_last=True)


            val_data_list = []
            for index, row in val.iterrows():
                val_data_list.append(smiles2graph(row, target_col))
            val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

            loaders.append([train_loader, val_loader, test_loader])

    else:
        for k, (train_index, test_index) in enumerate(
                cv_splitter.split(data)
        ):

            test = data.iloc[test_index]
            test_data_list = []
            for index, row in test.iterrows():
                test_data_list.append(smiles2graph(row, target_col))
            test_loader = DataLoader(test_data_list, num_workers=0, batch_size=batch_size)

            train_set = data.iloc[train_index]

            train, val = train_test_split(
                train_set,
                test_size=0.15,
                shuffle=True,
                random_state=seed
            )

            train_data_list = []
            for index, row in train.iterrows():
                train_data_list.append(smiles2graph(row, target_col))
            train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size, drop_last=True)

            val_data_list = []
            for index, row in val.iterrows():
                val_data_list.append(smiles2graph(row, target_col))
            val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

            loaders.append([train_loader, val_loader, test_loader])

    return loaders


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def create_loader(data, target_col, batch_size, **kwargs):
    data_list = []
    for index, row in data.iterrows():
        data_list.append(smiles2graph(row, target_col, **kwargs))

    data_loader = DataLoader(data_list, num_workers=0, batch_size=batch_size)

    return data_loader


def smiles2graph(data, target_col=None, **kwargs):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # smiles = smiles
    # y = withdrawn_col

    try:
        y = data[target_col]
    except:
        pass
    smiles = data['standardized_smiles']

    mol = Chem.MolFromSmiles(smiles)

    # atoms
    donor = []
    acceptor = []
    features = []
    names = []
    donor_string = []

    for atom in mol.GetAtoms():
        atom_feature_names = []
        atom_features = []
        atom_features += one_hot_vector(
            atom.GetAtomicNum(),
            [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
        )

        atom_feature_names.append(atom.GetSymbol())
        atom_features += one_hot_vector(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4]
        )
        atom_feature_names.append(atom.GetTotalNumHs())
        atom_features += one_hot_vector(
            atom.GetHybridization(),
            [HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
             HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.UNSPECIFIED]
        )
        atom_feature_names.append(atom.GetHybridization().__str__())

        atom_features.append(atom.IsInRing())
        atom_features.append(atom.GetIsAromatic())

        if atom.GetIsAromatic() == 1:
            atom_feature_names.append('Aromatic')
        else:
            atom_feature_names.append('Non-aromatic')

        if atom.IsInRing() == 1:
            atom_feature_names.append('Is in ring')
        else:
            atom_feature_names.append('Not in ring')

        donor.append(0)
        acceptor.append(0)

        donor_string.append('Not a donor or acceptor')

        atom_features = np.array(atom_features, dtype=int)
        atom_feature_names = np.array(atom_feature_names, dtype=object)
        features.append(atom_features)
        names.append(atom_feature_names)

    feats = factory.GetFeaturesForMol(mol)
    for j in range(0, len(feats)):
        if feats[j].GetFamily() == 'Donor':
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                donor[k] = 0
                donor_string[k] = 'Donor'
        elif feats[j].GetFamily() == 'Acceptor':
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                acceptor[k] = 1
                donor_string[k] = 'Acceptor'

    features = np.array(features, dtype=int)
    donor = np.array(donor, dtype=int)
    donor = donor[..., np.newaxis]
    acceptor = np.array(acceptor, dtype=int).transpose()
    acceptor = acceptor[..., np.newaxis]
    x = np.append(features, donor, axis=1)
    x = np.append(x, acceptor, axis=1)

    donor_string = np.array(donor_string, dtype=object)
    donor_string = donor_string[..., np.newaxis]

    names = np.array(names, dtype=object)
    names = np.append(names, donor_string, axis=1)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # add edges in both directions
            edges_list.append((i, j))
            edges_list.append((j, i))

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = Tensor(edge_index).long()
    graph['node_feat'] = Tensor(x)
    graph['feature_names'] = names
    try:
        graph['y'] = Tensor([y])
        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], y=graph['y'], feature_names=names)

    except:
        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], feature_names=names)
