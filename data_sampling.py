########
# Samples data from PubChem Library.
# Filters data depending on list of atoms to keep and characters (for example, SMILES having stereo information).
# Created by Rodrigo Hormazabal, 2022.
########

import pickle
from pathlib import Path

import pandas as pd
import pqdm
import wget
from pqdm.processes import pqdm
from termcolor import cprint
from tqdm import tqdm
from data_utils import get_mol_metadata

download_links = {'CID-SMILES': 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz',
                  'CID-InChI': 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-InChI.gz'}


class PubchemData:
    """
    Main class to handle loading, sampling and metadata generation for PubChemData.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = Path(cfg.data_path)
        self.dataset_name = cfg.dataset_name.lower()
        self.data_filename = f'CID-{cfg.dataset_name}.gz'
        self.data, self.metadata = None, None
        assert cfg.dataset_name in ['SMILES', 'InChI'], "Dataset type must one of ['SMILES', 'InChI']"

        if self.cfg.load_pickle:
            cprint(f'Loading filtered pickle file at {self.data_path / self.cfg.pickle_filename}...', 'green')
            self.metadata = pickle.load(open(self.data_path / self.cfg.pickle_filename, 'rb'))

        # if processed file does not exist download PubChem data
        if cfg.download or not (self.data_path / self.data_filename).exists():
            cprint('File does not exist or download flag set to True. Downloading...', 'red')
            wget.download(download_links[self.data_filename.replace('.gz', '')], out=str((self.data_path / self.data_filename).absolute()))

    def load_pubchem_dataset(self) -> None:
        """
        Loads data from downloaded pubchem zip file.
        :return: None.
        """
        cprint('Loading PubChem data...', 'yellow')
        self.data = pd.read_csv(self.data_path / self.data_filename, compression='gzip', sep='\t', nrows=self.cfg.pubchem_nrows)
        self.data.columns = ['cid', self.dataset_name]

    def filter_molecules_by_sequence(self, sequence_filters) -> None:
        """
        Applies filters to raw PubChem dataset.
        Filters: Characters to keep / drop, atoms to keep, numbers of characters.
        :return: None.
        """
        cprint(f'Filtering data | Keep: {sequence_filters.chars_to_keep}', 'blue')
        cprint(f'Drop: {sequence_filters.chars_to_drop}', 'blue')
        cprint(f'Max len: {sequence_filters.max_len}...', 'blue')

        # keep / drop sequences containing
        self.data = self.data[(self.data.smiles.str.contains('|'.join(sequence_filters.chars_to_keep))) &
                              ~(self.data.smiles.str.contains('|'.join(sequence_filters.chars_to_drop)))]

        # atoms to keep
        self.data = self.data[self.data.smiles.str.contains('|'.join(sequence_filters.atoms_to_keep))]

        # number of characters filter
        self.data = self.data[(self.data.smiles.str.len() < sequence_filters.max_len) &
                              (self.data.smiles.str.len() > sequence_filters.min_len)]
        cprint(f'Molecules after filtering: {self.data.shape[0]}', 'blue')

    def filter_molecules_by_metadata(self, metadata_filters) -> None:
        """
        Filter molecules through metadata (e.g. number of atoms, number of bonds, etc.)
        :return: None.
        """
        cprint(f'Filtering by metadata | Max atoms: {metadata_filters.max_atoms}', 'blue')
        self.metadata = self.metadata[self.metadata.n_atoms <= metadata_filters.max_atoms]
        self.data = self.data[self.data.cid.isin(self.metadata.cid)]
        cprint(f'Molecules after filtering: {self.metadata.shape[0]}', 'blue')

    def generate_metadata(self) -> None:
        """
        Generates metadata for each molecule and stores it in self.metadata.
        In case of error with SMILES/metadata, drop molecule from self.data.
        :return: None.
        """
        # get metadata for each molecule
        cprint(f'Generating metadata...', 'yellow')
        metadata = pqdm(self.data.smiles, get_mol_metadata, n_jobs=self.cfg.n_jobs)

        # drop molecules with corrupted SMILES/metadata
        metadata = [m for m in metadata if len(m) != 0]
        cprint(f'Molecules with metadata: {len(metadata)}...', 'blue')

        # keep PubChem CID
        self.metadata = pd.DataFrame(metadata)
        self.metadata = pd.merge(self.data[['cid', self.dataset_name]], self.metadata, how='inner', left_on=self.dataset_name, right_on=self.dataset_name)
        self.data = self.data[self.data.cid.isin(self.metadata.cid)]
        with open(self.data_path / self.cfg.pickle_filename, 'wb') as f:
            pickle.dump(self.metadata, f)

    def get_dataset_smarts_counts(self) -> dict:
        """
        Returns dictionary of SMARTS counts for each molecule.
        :return: dict of SMARTS counts.
        """
        # counting atom distribution
        total_counts = dict()
        for idx, row in tqdm(self.metadata.iterrows(), desc='Calculating SMARTS token distribution...'):
            for k, v in row.unique_smarts.items():
                if k not in total_counts.keys():
                    total_counts[k] = v
                else:
                    total_counts[k] += v
        return total_counts
