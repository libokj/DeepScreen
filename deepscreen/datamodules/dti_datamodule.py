from itertools import permutations
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from deepscreen.utils.utils import convert_y_unit, drug2enc, protein2enc


class DTIDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            dataset_name: str,
            drug_featurizer: callable,
            protein_featurizer: callable,
            binarization: bool = False,
            threshold: float = 30,
            logarithmic: bool = False,
    ):
        # affinity = pd.read_csv(f'{data_dir}{dataset_name}/affinity.txt', header=None, sep=' ')
        # with open(f'{data_dir}{dataset_name}/fasta.txt') as f:
        #     target = list(json.load(f).values())
        # with open(f'{data_dir}{dataset_name}/SMILES.txt') as f:
        #     drug = list(json.load(f).values())

        # for i in range(len(drug)):
        #     for j in range(len(target)):
        #         smiles.append(drug[i])
        #         fasta.append(target[j])
        #         y.append(affinity.values[i, j])
        df = pd.read_csv(f'{data_dir}{dataset_name}.csv', header=0, sep=',',
                         dtype={'X1': str, 'X2': str})

        smiles = df['X1']
        fasta = df['X2']

        self.smiles, self.fasta = np.array(smiles), np.array(fasta),
        self.drug_featurizer, self.protein_featurizer = drug2enc, protein2enc

        if 'Y' in df:
            y = df['Y']
            if binarization:
                y = [1 if i else 0 for i in np.array(y) < threshold]
            elif logarithmic:
                y = convert_y_unit(np.array(y), 'nM', 'p')
            self.y = np.array(y)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if hasattr(self, 'y'):
            return self.drug_featurizer(self.smiles[idx]), self.protein_featurizer(self.fasta[idx]), self.y[idx]
        else:
            return self.drug_featurizer(self.smiles[idx]), self.protein_featurizer(self.fasta[idx])


class DTIDataModule(LightningDataModule):
    """
    DAVIS DataModule

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            drug_featurizer: callable,
            protein_featurizer: callable,
            batch_size: int = 128,
            num_workers: int = 0,
            pin_memory: bool = False,
            binarization: bool = False,
            threshold: float = 30,
            logarithmic: bool = True,
            shuffle: bool = False,
            drop_last: bool = False,
            data_dir: str = "data/",
            dataset_name: Optional[str] = None,
            predict: bool = False,
            train_val_test_split: Optional[Tuple[float]] = (0.7, 0.1, 0.2),
            data_split: Optional[callable] = random_split,
            train_dataset: Optional[Dataset] = None,
            val_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            predict_dataset: Optional[Dataset] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), ]
        )
        if dataset_name and any([train_dataset, val_dataset, test_dataset]):
            raise ValueError("Please do not provide train_dataset, val_dataset, and test_dataset "
                             "after specifying dataset_name.")
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset
        self.data_predict = predict_dataset

        if predict:
            self.data_predict = DTIDataset(data_dir=self.hparams.data_dir,
                                           drug_featurizer=self.hparams.drug_featurizer,
                                           protein_featurizer=self.hparams.protein_featurizer,
                                           dataset_name=self.hparams.dataset_name,
                                           binarization=self.hparams.binarization,
                                           threshold=self.hparams.threshold,
                                           logarithmic=self.hparams.logarithmic)

    def prepare_data(self):
        """
        Download data if needed.
        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None, encoding: str = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded in initialization
        if not any([self.data_train, self.data_val, self.data_test, self.data_predict]):
            dataset = DTIDataset(data_dir=self.hparams.data_dir,
                                 drug_featurizer=self.hparams.drug_featurizer,
                                 protein_featurizer=self.hparams.protein_featurizer,
                                 dataset_name=self.hparams.dataset_name,
                                 binarization=self.hparams.binarization,
                                 threshold=self.hparams.threshold,
                                 logarithmic=self.hparams.logarithmic)
            self.data_train, self.data_val, self.data_test = self.hparams.data_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "dti.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
