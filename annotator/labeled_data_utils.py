import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.patches import Rectangle
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm

from annotator.data_utils import get_mol_metadata

molfile_extensions = {'JPO': 'sdf',
                      'USPTO': 'MOL',
                      'UOB': 'mol',
                      'CLEF': 'mol'}

color_map = {'[C@@H]': (1, 0, 1),
             '[C@H]': (1, 1, 0),
             '[C@]': (1, 0.2, 0.4),
             '[C@@]': (0.2, 0.8, 0.1)}

valence_map = {0: '',
               1: 'H',
               2: 'H2',
               3: 'H3'}


def draw_mol_with_smarts_annotations(mol, filename):
    """
    Draw molecule with SMARTS annotations for crosschecking.
    :param mol: rdkit molecule. [rdkit.Chem.rdchem.Mol]
    :param filename: filename to save the image. [str]
    :return: None.
    """
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)

    highlight_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSmarts() in ['[C@@H]', '[C@H]', '[C@]', '[C@@]']]
    highlight_colors = {atom.GetIdx(): color_map[atom.GetSmarts()] for atom in mol.GetAtoms() if atom.GetSmarts() in ['[C@@H]', '[C@H]', '[C@]', '[C@@]']}

    rdMolDraw2D.PrepareAndDrawMolecule(d,
                                       mol,
                                       highlightAtoms=highlight_idxs,
                                       highlightAtomColors=highlight_colors)
    for atom in mol.GetAtoms():
        smarts = f"[{atom.GetSmarts()}{valence_map[atom.GetImplicitValence()]}]"
        if atom.GetIsAromatic():
            smarts = smarts.replace(']', '_arom]')
        smarts = smarts.replace('[[', '[').replace(']]', ']')
        atom.SetProp('atomNote', smarts)

    for bond in mol.GetBonds():
        bond_stereo = bond.GetStereo().name
        if bond_stereo != 'STEREONONE':
            bond.SetProp('bondNote', f'[{bond_stereo}]')

    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText(filename)
    return


class LabeledData:
    def __init__(self, root_path, labeled_data_path, gt_data_path, dataset_name):
        """
        Main class to save, edit and visualize annotations.
        :param root_path: Root path to the dataset. [str, Path]
        :param labeled_data_path: Path to labeled data. [str, Path]
        :param gt_data_path: Path to ground truth data. [str, Path]
        :param dataset_name: Name of the dataset. [str]
        """
        self.root_path = Path(root_path)
        labeled_data = json.load(open(self.root_path / labeled_data_path, 'r'))
        self.dataset_name = dataset_name
        self.images = pd.DataFrame(labeled_data['images']).set_index('file_name')
        self.categories = pd.DataFrame(labeled_data['categories'])
        self.annotations = pd.DataFrame(labeled_data['annotations'])

        # add smarts column to annotations
        id_to_smarts_map = self.categories[['id', 'name']].set_index('id').to_dict()['name']
        self.annotations['smarts'] = self.annotations.category_id.map(id_to_smarts_map)

        # bond and non-bond ids
        self.bond_smarts = self.categories[self.categories.supercategory == 'bond'].name
        self.nonbond_smarts = self.categories[self.categories.supercategory != 'bond'].name

        self.image_id_mapping = None
        self.annotations_id_mapping = None
        self.category_id_mapping = None

        # plotting info
        self.instance_colors = {k: tuple(np.random.random(size=3).tolist()) for k in self.annotations.smarts.unique()}

        # load gt data
        if 'molfile_path' not in self.images.columns:
            self.images['molfile_name'] = self.images.index.str.replace('png', molfile_extensions[dataset_name])
            mols = self.images.molfile_name.apply(lambda x: Chem.MolFromMolFile(str(self.root_path / gt_data_path / x)))
            self.metadata = pd.DataFrame([get_mol_metadata(Chem.MolToSmiles(m)) for m in mols], index=mols.index)

    def plot_predictions_from_labels(self, filepath, annotations, metadata, save_fig=False, output_dir=None, ticks=False):
        """
        Plot predictions from annotation files.
        :param filepath: Path to image file. Can be a string or a Path object. [str, Path]
        :param annotations: Annotations dataframe. [pd.DataFrame]
        :param metadata: Metadata dataframe. [pd.DataFrame]
        :param save_fig: Whether to save the figure. [bool]
        :param output_dir: Output directory. [str, Path]
        :param ticks: Whether to show ticks. [bool]
        :return: None
        """
        # load image
        filepath = Path(filepath)
        img = plt.imread(filepath)
        dpi = 100

        bond_annotations = annotations[annotations.smarts.isin(self.bond_smarts)]
        nonbond_annotations = annotations[annotations.smarts.isin(self.nonbond_smarts)]

        # plot non-bonds and bonds separately for better visualization
        for anns_type, anns in zip(['atom', 'bond'], [nonbond_annotations, bond_annotations]):

            fig = plt.figure(frameon=False)
            fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
            current_axis = plt.Axes(fig, [0., 0., 1., 1.])
            current_axis.set_axis_off()
            fig.add_axes(current_axis)
            current_axis.imshow(img)

            if ticks:
                plt.grid(axis='both')
                current_axis.xaxis.set_major_locator(ticker.MultipleLocator(ticks))
                current_axis.yaxis.set_major_locator(ticker.MultipleLocator(ticks))

            for idx, row in anns.iterrows():
                # rectangle needs width and height
                x1, y1, w, h = row.bbox

                # color rectangles and add annotation type as text
                current_axis.add_patch(Rectangle((x1, y1), w, h, fill=None, alpha=1, color=self.instance_colors[row.smarts]))
                plt.plot(x1 + w / 2, y1 + h / 2, 'rx')
                plt.text(x1, y1, row.smarts, fontsize=10, color=self.instance_colors[row.smarts])

            if save_fig:
                output_dir = Path(output_dir)
                image_name = filepath.stem + f'_{anns_type}_annotations.png'
                plt.savefig(str(output_dir / image_name), dpi=dpi)
                draw_mol_with_smarts_annotations(Chem.MolFromSmiles(metadata.SMILES),
                                                 str(output_dir / f'{filepath.stem}_rdkit_smarts.png'))
                plt.clf()

            else:
                plt.show()
                plt.clf()

    def all_annotations_to_img(self, output_dir, ticks):
        """
        Plot all annotations with the corresponding images for the dataset.
        :param output_dir: directory to save the images. [str]
        :param ticks: Ticks for the grid. [int]
        :return: None.
        """
        for image_name, row in tqdm(self.images.iterrows(), desc='Generating annotation images for verification...'):
            image_annotations = self.annotations[self.annotations.image_id == row.id]
            image_metadata = self.metadata.loc[image_name]
            self.plot_predictions_from_labels(self.root_path / f'./data/images/{self.dataset_name}/{image_name}',
                                              image_annotations,
                                              image_metadata,
                                              save_fig=True,
                                              output_dir=self.root_path / output_dir,
                                              ticks=ticks)

    def save_to_coco_annotator_format(self, annotations_filepath, zip_images_filepath):
        """
        Export annotation data in coco format for coco-annotator.
        :param annotations_filepath:
        :param zip_images_filepath:
        :return:
        """
        # json data for annotator
        images_columns = ['id', 'file_name', 'width', 'height']
        annotations_columns = ['smarts', 'image_id', 'iscrowd', 'bbox', 'area', 'segmentation', 'category_id', 'id']

        data = {'images': self.images.reset_index()[images_columns].to_dict(orient='records'),
                'annotations': self.annotations[annotations_columns].to_dict(orient='records'),
                'categories': self.categories.to_dict(orient='records')}
        json.dump(data, open(annotations_filepath, 'w'))
        print(f"Annotations saved at: {annotations_filepath}")

        # zipped images
        assert self.dataset_name in zip_images_filepath, "Provided paths and dataset name do not match."

        # path to folder which needs to be zipped
        directory = f'./data/images/{self.dataset_name}/'

        # writing files to a zipfile
        with ZipFile(zip_images_filepath, 'w') as zip_file:
            # writing each file one by one
            for file in self.images.index:
                zip_file.write(f'{directory}{file}')

        print(f'Zipped images saved at in path: {zip_images_filepath}')
        return
