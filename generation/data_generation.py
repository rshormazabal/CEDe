########
# Generates synthetic chemical images with bounding box annotations for chemical entities.
# Chemical entities include: Atoms, bonds, pseudoatoms, etc.
# Created by Rodrigo Hormazabal, 2022.
########
import os
import random
import re
import string
import warnings
from os import environ
from pathlib import Path

import numpy as np
import pandas as pd
from indigo import Indigo
from indigo.renderer import IndigoRenderer
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import Draw, rdmolops
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from tqdm import tqdm

from annotator.data_utils import bond_type_mappings, get_svg_instances

warnings.filterwarnings(action='ignore')
indigo_engine = Indigo()
rend = IndigoRenderer(indigo_engine)
indigo_engine.setOption("render-image-height", 768)
indigo_engine.setOption("render-image-width", 768)


def pseudoatom_possible_idx(smiles):
    """
    Get posible atom indexes where to assign pseudoatoms. Get number of atoms with degree equal to 1 and a single bond.
    :param smiles:
    :return:
    """
    mol = rdmolops.AddHs(Chem.MolFromSmiles(smiles), explicitOnly=True)
    smarts = Chem.MolFromSmarts('[*D1]-*')
    return [s[0] for s in mol.GetSubstructMatches(smarts)]


def pseudoatom_modification(smiles, pseudoatom):
    """

    :param smiles:
    :param pseudoatom:
    :return:
    """

    # get mol object and chose possible positions to substitute for pseudo atoms
    mol = rdmolops.AddHs(Chem.MolFromSmiles(smiles), explicitOnly=True)
    candidates_list = pseudoatom_possible_idx(smiles)

    # choose random idx and add pseudo character at idx
    idx = random.choice(candidates_list)
    ed_mol = Chem.rdchem.EditableMol(mol)
    ed_mol.ReplaceAtom(idx, Chem.Atom('*'))
    mol_re = ed_mol.GetMol()

    # read new mol object and reconstruct smiles -> mol object
    cxsmiles = Chem.MolToSmiles(mol_re)
    mol_re_rearranged = Chem.MolFromSmiles(cxsmiles)
    mol_re_rearranged = rdmolops.AddHs(mol_re_rearranged, explicitOnly=True)

    # get new idx for subtituted atom
    for idx, atom in enumerate(mol_re_rearranged.GetAtoms()):
        if atom.GetSymbol() == '*':
            break

    # generate new cxsmiles string
    num_atoms = mol_re_rearranged.GetNumAtoms()
    cxsmiles = cxsmiles + f'   |${";" * idx}{pseudoatom}{";" * (num_atoms - 1 - idx)}$|'
    return cxsmiles, idx


def replace_atom_to_pseudoatom(molfile, pseudoatom, pseudo_index):
    """
    Edit file on disk to account for added pseudoatoms.
    :param molfile:
    :param pseudoatom:
    :param pseudo_index:
    :return:
    """
    # open file
    strings = [_ for _ in open(molfile, 'r').readlines()]
    pseudoatom_pos = 7 + pseudo_index
    strings[pseudoatom_pos] = ' '.join(strings[pseudoatom_pos].split(' ')[:4] + [pseudoatom] + strings[pseudoatom_pos].split(' ')[5:])

    file2 = open(molfile, "w+")
    for L in strings:
        file2.writelines(L)
    file2.close()
    return


def convert_smiles_with_augmentations_to_svg(smiles: str, augmentations: dict, pseudo_orientation='right') -> (str, str):
    """

    :param smiles: SMILES string. [str]
    :param augmentations: Augmentation dictionary. [dict]
    :param pseudo_orientation:
    :return: SVG, CXSMILES. [str, str]
    """
    assert pseudo_orientation in ['right', 'left'], 'Pseudoatom orientation must be either right or left.'
    if augmentations['pseudo_replace']:
        smiles, index_pseudo = pseudoatom_modification(smiles, augmentations['pseudo'][pseudo_orientation])
        augmentations['pseudo_index'] = index_pseudo
        mol = rdmolops.AddHs(Chem.MolFromSmiles(smiles), explicitOnly=True)
    else:
        mol_tmp = rdmolops.AddHs(Chem.MolFromSmiles(smiles), explicitOnly=True)
        mol = transform_mol(mol_tmp, augmentations)

    rdbase = environ['RDBASE']

    dm = Draw.PrepareMolForDrawing(mol, forceCoords=False)
    d2d = Draw.MolDraw2DSVG(augmentations['img_size'], augmentations['img_size'])
    d2d.drawOptions().minFontSize = augmentations['font_size']
    d2d.drawOptions().maxFontSize = -1
    d2d.drawOptions().bondLineWidth = augmentations['linewidth']
    d2d.drawOptions().fontFile = augmentations['font_type'].format(rdbase)
    d2d.drawOptions().rotate = augmentations['rotation_angle']
    d2d.drawOptions().additionalAtomLabelPadding = 0.1
    d2d.DrawMolecule(dm)
    d2d.TagAtoms(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return smiles, svg


def transform_mol(mol, augmentations: dict):
    """

    :param mol:
    :param augmentations:
    :return:
    """
    n_atoms = mol.GetNumAtoms()
    molblock = Chem.MolToMolBlock(mol)

    molblock_list = molblock.split('\n')

    for i in range(n_atoms):
        tmp = molblock_list[4 + i].split()
        tmp[1] = str(round(float(tmp[1]) * augmentations['xy_sheer'], 4))
        tmp[1] = ' ' * (3 - len(tmp[1].split('.')[0])) + tmp[1] + '0' * (4 - len(tmp[1].split('.')[1])) + '  '
        tmp[0] = ' ' * (5 - len(tmp[0].split('.')[0])) + tmp[0] + '0' * (4 - len(tmp[0].split('.')[1]))
        tmp[3] = tmp[3] + ' ' * (2 - len(tmp[3]))
        molblock_list[4 + i] = '  '.join(tmp[:3]) + ' ' + '  '.join(tmp[3:])
    molblock_list_reconstruct = '\n'.join(molblock_list)
    return Chem.MolFromMolBlock(molblock_list_reconstruct)


def get_annotations(smiles: str, smiles_pubchem: str, augmentations: dict, bbox_params: dict) -> dict:
    """
    Create json annotation file in COCO-format.
    :param smiles: Reconstructed compounds' SMILE string with RDKit. [str]
    :param smiles_pubchem: Compounds' SMILE string from PubChem. [str]
    :param augmentations: Data generation sampled augmentations. [dict]
    :param bbox_params: Bounding box parameters. [dict]
    :return: Dict with chemical instances' bouding box annotations and metadata. [dict]
    """
    # create augmented molecule and get SVG string
    pseudo_orientation = 'right'
    smiles_mod, svg = convert_smiles_with_augmentations_to_svg(smiles, augmentations, pseudo_orientation=pseudo_orientation)

    # create mol object
    mol = rdmolops.AddHs(Chem.MolFromSmiles(smiles_mod), explicitOnly=True)

    # get atom masks
    atom_instances, bond_instances = get_svg_instances(mol, svg, bbox_params, split_bonds_atoms=True)

    # in case the pseudoatom orientation change depending on position, correct it.
    pseudoatoms = augmentations['pseudo']
    if augmentations['pseudo_replace'] and (pseudoatoms['right'] != pseudoatoms['left']):
        # get pseudoatom bond direction to decide on which side to place orientation to place pseudoatom
        pseudoatom_bond_atoms = bond_instances.loc[bond_instances.instance.str.contains(f'atom-{augmentations["pseudo_index"]}'), 'connecting_nodes'].values
        pseudatom_connected_atom = [atom_idx for atom_idx in pseudoatom_bond_atoms[0] if atom_idx != augmentations['pseudo_index']]

        pseudoatom_box = atom_instances[atom_instances.instance == f'atom-{augmentations["pseudo_index"]}']
        pseudoatom_box = pseudoatom_box[['x_min', 'x_max', 'y_min', 'y_max']].values[0]

        connected_atom_box = atom_instances[atom_instances.instance == f'atom-{pseudatom_connected_atom[0]}']
        connected_atom_box = connected_atom_box[['x_min', 'x_max', 'y_min', 'y_max']].values[0]

        # get the x center of intersection box
        x_min = max(pseudoatom_box[0], connected_atom_box[0])
        x_max = min(pseudoatom_box[1], connected_atom_box[1])
        intersection_box_x_center = (x_min + x_max) / 2

        # calculate distance from x center of intersection box to left and right side of pseudoatom box
        pseudoatom_box_center_to_left = abs(intersection_box_x_center - pseudoatom_box[0])
        pseudoatom_box_center_to_right = abs(intersection_box_x_center - pseudoatom_box[1])

        # set orientation to left if incoming bonds is closer to right edge of pseudoatom box
        if pseudoatom_box_center_to_right < pseudoatom_box_center_to_left:
            pseudo_orientation = 'left'
            smiles_mod, svg = convert_smiles_with_augmentations_to_svg(smiles, augmentations, pseudo_orientation=pseudo_orientation)

            # create mol object
            mol = rdmolops.AddHs(Chem.MolFromSmiles(smiles_mod), explicitOnly=True)

            # recalculate bouding boxes
            atom_instances, bond_instances = get_svg_instances(mol, svg, bbox_params, split_bonds_atoms=True)

    # image meta data
    data = {'cxsmiles': smiles_mod,
            'smiles': smiles_pubchem,
            'n_atoms': mol.GetNumAtoms(),
            'n_bonds': mol.GetNumBonds(),
            'pseudo_orientation': pseudo_orientation if pseudo_orientation else None}

    # get bond types from mol3k block
    v3k_mol_block = Chem.MolToV3KMolBlock(mol).replace('\n', 'lj')
    bond_directions = [bond_dir.split() for bond_dir in re.findall("\d* \d* CFG=\d*", v3k_mol_block)]
    bond_directions = {f'atom-{int(b1) - 1} atom-{int(b2) - 1}': bond_type_mappings[direction] for b1, b2, direction in bond_directions}

    # instance labels
    pseudoatom_smarts = f'[{pseudoatoms["right"]}]'
    atom_info_keys = ['element', 'charge', 'aromaticity', 'isotope', 'implicitHs']
    for idx, instance in atom_instances.iterrows():
        # pseudatoms
        if instance.instance_smarts == '[*]':
            atom_instances.loc[idx, atom_info_keys] = ['pseudoatom', 0, False, 0, 0]
            atom_instances.loc[idx, 'instance_smarts'] = pseudoatom_smarts
        # others
        else:
            smarts = Chem.MolFromSmarts(instance.instance_smarts).GetAtoms()[0]
            atom_instances.loc[idx, atom_info_keys] = [smarts.GetSymbol(),
                                                       smarts.GetFormalCharge(),
                                                       smarts.GetIsAromatic(),
                                                       smarts.GetIsotope(),
                                                       smarts.GetNumExplicitHs()]
    for idx, instance in bond_instances.iterrows():
        # get connected atoms, ignoring the bond numbering
        connected_atoms = instance.instance.split(' ')[1:]
        connected_atoms = ' '.join(connected_atoms)
        if connected_atoms in bond_directions.keys():
            bond_instances.loc[idx, 'instance_smarts'] = bond_directions[connected_atoms]

    data['annotations'] = atom_instances.to_dict(orient='records')
    data['annotations'].extend(bond_instances.to_dict(orient='records'))
    data['svg'] = svg

    return data


def generate_sample(cid: int, reconstructed_smiles: str, smiles: str, augmentations: dict, bbox_params: dict) -> dict:
    """
    Generate images, annotations and metadata for an SMILES sample with sampled augmentations.
    :param cid: Pubchem CID of the compound. [int]
    :param reconstructed_smiles: RDKit reconstructed SMILES string. [str]
    :param smiles: Pubchem database SMILES string. [str]
    :param augmentations: Data generation sampled augmentations. [dict]
    :param bbox_params: Bounding box parameters. [dict]
    :return: Dictionary with image, annotations and metadata. [dict]
    """

    # Create path if it doesnt exist
    rdkit_path = Path(f'./data/generated_data/rdkit/{str(cid)[:5]}')
    indigo_path = Path(f'./data/generated_data/indigo/{str(cid)[:5]}')
    rdkit_path.mkdir(parents=True, exist_ok=True)
    indigo_path.mkdir(parents=True, exist_ok=True)

    # check whether an atom for pseudoatom replacement exist in the compound
    augmentations['pseudo_replace'] = False if len(pseudoatom_possible_idx(reconstructed_smiles)) == 0 else augmentations['pseudo_replace']

    # get annotations and save ID number from PubChem CID
    data = get_annotations(reconstructed_smiles, smiles, augmentations, bbox_params)
    data['height'], data['width'] = augmentations['img_size'], augmentations['img_size']
    data['id'] = cid

    # save indigo file
    mol = indigo_engine.loadMolecule(data['cxsmiles'])
    rend.renderToFile(mol, f'{indigo_path.resolve()}/{cid}.png')

    # save molfile
    mol = rdmolops.AddHs(Chem.MolFromSmiles(smiles), explicitOnly=True)
    Chem.MolToV3KMolFile(mol, f'{rdkit_path.resolve()}/{cid}.mol')
    if augmentations['pseudo_replace']:
        replace_atom_to_pseudoatom(f'{rdkit_path.resolve()}/{cid}.mol',
                                   augmentations['pseudo'][data['pseudo_orientation']],
                                   augmentations['pseudo_index'])

    # save svg and png files
    save_image_files(data['svg'], f'{rdkit_path.resolve()}/{cid}')
    return data


def save_image_files(svg, filename):
    """
    Save svg and png files from svg string.
    :param svg: SVG string. [str]
    :param filename: Filename without extension. [str]
    :return: None
    """
    strings = [line for line in svg.split('\n') if 'selector' not in line]
    for i in range(len(strings)):
        if '#' in strings[i]:
            index_n = strings[i].index('#')
            index_nr = strings[i].rindex('#')
            if strings[i][index_n:index_n + 7] != '#FFFFFF':
                strings[i] = strings[i][:index_n] + '#000000' + strings[i][index_n + 7:]
                strings[i] = strings[i][:index_nr] + '#000000' + strings[i][index_nr + 7:]
        if 'stroke-linecap' in strings[i]:
            strings[i] = ';'.join(strings[i].split(';')[:4] + ['stroke-linecap:round'] + strings[10].split(';')[5:])

    svg = '\n'.join(strings)

    # write svg file
    write_svg = open(f'{filename}.svg', "wt")
    write_svg.write(svg)
    write_svg.close()
    drawing = svg2rlg(f'{filename}.svg')

    # write png file
    renderPM.drawToFile(drawing, f'{filename}.png', fmt="PNG")
    return


class Augmentations:
    def __init__(self,
                 img_size,
                 pseudo_prob,
                 pseudo_type,
                 pseudoatoms_lib_path,
                 aug_sample_params,
                 seed=1212):
        """

        :param img_size:
        :param pseudo_prob:
        :param pseudo_type:
        :param pseudoatoms_lib_path:
        """
        self.img_size = img_size
        self.pseudo_prob = pseudo_prob
        self.pseudo_type = pseudo_type
        self.pseudoatoms_lib = pd.read_csv(f'.{pseudoatoms_lib_path}')
        self.seed = seed
        self.aug_sample_params = aug_sample_params

        self.fonts_path = aug_sample_params.fonts_path
        # get font paths
        self.fonts_list = [f for f in os.listdir(aug_sample_params.fonts_path) if f.endswith('.ttf')]

    def sample_augmentation(self):
        """
        Random sampling of augmentation methods
        :return: Sampled augmentations. [dict]
        """

        # randomly select augmentation parameters
        linewidth = np.random.randint(*self.aug_sample_params.linewidth_range)
        font_size = np.random.randint(*self.aug_sample_params.font_size_range)
        font_type_idx = np.random.randint(0, len(self.fonts_list))
        rotation_angle = np.random.randint(*self.aug_sample_params.rotation_angle_range)
        xy_sheer = round(np.random.uniform(*self.aug_sample_params.xy_sheer_range), 1)
        pseudo_replace = np.random.uniform() > 1 - self.pseudo_prob
        pseudo = {'right': None, 'left': None}

        if pseudo_replace:
            if self.pseudo_type == 'R':
                pseudo = 'R' + str(np.random.randint(1, 10))
            elif self.pseudo_type == 'random':
                letters_cha = string.ascii_letters
                letters_num = string.digits
                pseudo = ''.join([random.choice(letters_cha) for _ in range(3)] + [random.choice(letters_num) for _ in range(2)])
            elif self.pseudo_type == 'given':
                pseudo = self.pseudoatoms_lib.sample()
                pseudo = pseudo[['left', 'right']].to_dict(orient='records')[0]

        # augmentation parameters for the generated sample
        augmentations = {'img_size': self.img_size,
                         'linewidth': linewidth,
                         'font_size': font_size,
                         'font_scale': 0.75,
                         'font_type': self.fonts_path + self.fonts_list[font_type_idx],
                         'rotation_angle': rotation_angle,
                         'xy_sheer': xy_sheer,
                         'pseudo_index': 0,
                         'pseudo_replace': pseudo_replace,
                         'pseudo': pseudo}
        return augmentations


def unique_atoms(mol: Chem.Mol) -> set[str]:
    """
    Get unique atoms in a molecule (not including pseudo atoms).
    :param mol: Rdkit molecule object. [Chem.Mol]
    :return: unique atoms in a molecule. [set]
    """
    unique: set[str] = set()
    for atom in mol.GetAtoms():
        unique.add(atom.GetSymbol())
    return unique


def transform_annotations_to_coco_style(data: list, coco_dataset_metadata: DictConfig) -> dict:
    """
    Transform raw annotations to COCO style. Drops SVG string data from annotations dicts.
    :param data: Synthetic annotations' data in raw format. [list]
    :param coco_dataset_metadata: COCO dataset metadata. [DictConfig]
    :return: Synthetic annotations' data in COCO style. [list]
    """
    coco_dataset = {'info': {'description': coco_dataset_metadata.description,
                             'url': coco_dataset_metadata.url,
                             'version': coco_dataset_metadata.version,
                             'year': coco_dataset_metadata.year,
                             'contributor': coco_dataset_metadata.contributor,
                             'creation_date': coco_dataset_metadata.creation_date,
                             'licences': {"url": coco_dataset_metadata.license_url,
                                          "id": 1,
                                          "name": coco_dataset_metadata.license_name}
                             },
                    'images': [],
                    'annotations': []}

    categories = dict()
    annotation_idx = 0
    # iterate over all generated samples
    for idx, image_sample in tqdm(enumerate(data)):
        # pop SVG (too heavy)
        image_sample.pop('svg', None)

        # get annotations and pop from original dictionary
        annotations = image_sample.pop('annotations')

        # image data
        image_sample['file_name'] = f'./data/generated_data/rdkit/{str(image_sample["id"])[:5]}/{image_sample["id"]}.png'
        image_sample['pubchem_cid'] = image_sample['id']
        image_sample['id'] = idx
        image_sample['license'] = 1
        coco_dataset['images'].append(image_sample)

        # connectivity information
        atom_local_to_global_id = dict()
        image_sample['connectivity'] = []
        for annotation in annotations:
            # if category is not in list of categories, add it.
            if annotation['instance_smarts'] not in categories:
                categories[annotation['instance_smarts']] = len(categories) + 1

            # add necesary info to match COCO style annotations
            annotation['image_id'] = image_sample['id']
            annotation['iscrowd'] = 0

            # bounding box
            annotation['bbox'] = [annotation['x_min'],
                                  annotation['y_min'],
                                  round(annotation['x_max'] - annotation['x_min'], 2),
                                  round(annotation['y_max'] - annotation['y_min'], 2)]

            if annotation['instance_type'] == 'atom':
                atom_local_to_global_id[int(annotation['instance'].split('-')[-1])] = annotation_idx

            if annotation['instance_type'] == 'bond':
                # map local atom ID to global instance ID
                connected_nodes = annotation['connecting_nodes']
                connected_nodes = (atom_local_to_global_id[connected_nodes[0]],
                                   atom_local_to_global_id[connected_nodes[1]])
                image_sample['connectivity'].append(connected_nodes)
                annotation['connecting_nodes'] = connected_nodes

            for k in ['x_min', 'y_min', 'x_max', 'y_max', 'instance_type', 'instance']:
                annotation.pop(k, None)

            # area in pxls
            annotation['area'] = round(annotation['bbox'][2] * annotation['bbox'][3], 3)

            # category id and annotationotation id
            annotation['category_id'] = categories[annotation['instance_smarts']]
            annotation['id'] = annotation_idx
            annotation_idx += 1

            # append to annotationotations
            coco_dataset['annotations'].append(annotation)

    # match categories format
    coco_dataset['categories'] = [{'id': v, 'name': k, 'supercategory': 'atom'} for k, v in categories.items()]
    return coco_dataset
