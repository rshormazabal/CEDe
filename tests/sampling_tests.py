import unittest
from hydra import compose, initialize
from data_sampling import PubchemData


class SamplerTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # loading configuration file
        initialize(config_path="conf", job_name="test_cases", version_base="1.2")
        cls.cfg = compose(config_name="test_pubchem_sampling")

    @classmethod
    def setUp(cls) -> None:
        # test dataset loading
        pubchem_data = PubchemData(cls.cfg)
        pubchem_data.load_pubchem_dataset()
        pubchem_data.generate_metadata()
        cls.pubchem_data = pubchem_data

    def test_smarts_counts(self):
        """
        Test for getting smarts counts method of PubchemData class.
        :return: None.
        """
        # test by smarts counts of filteres data
        smarts_counts = self.pubchem_data.get_dataset_smarts_counts()
        self.assertEqual(smarts_counts['-'], 190, 'Number of single bond instances didn\'t match expected value.')
        self.assertEqual(smarts_counts['='], 50, 'Number of double bond instances didn\'t match expected value.')

    def test_filter_by_sequence(self):
        """
        Test for filtering by sequence method of PubchemData class.
        :return: None.
        """
        # test filtering by sequence
        self.pubchem_data.filter_molecules_by_sequence(self.cfg.sequence_filters)
        self.assertEqual(self.pubchem_data.data.shape[0], 2, 'Number of filtered molecules didn\'t match expected value.')

    def test_filter_by_metadata(self):
        """
        Test for filtering by metadata method of PubchemData class.
        :return: None.
        """
        # test filtering by metadata
        self.pubchem_data.filter_molecules_by_metadata(self.cfg.metadata_filters)
        smarts_counts = self.pubchem_data.get_dataset_smarts_counts()
        self.assertEqual(smarts_counts['-'], 169, 'Number of single bond instances didn\'t match expected value.')
        self.assertEqual(smarts_counts['='], 46, 'Number of double bond instances didn\'t match expected value.')
        self.assertEqual(smarts_counts['[C]'], 26, 'Number of [C] instances didn\'t match expected value.')
        self.assertEqual(smarts_counts['[Cl]'], 6, 'Number of [Cl] instances didn\'t match expected value.')
