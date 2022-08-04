import re
from collections import Counter
from io import BytesIO
from typing import Union
from xml.dom import minidom

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem import rdmolops, Draw

# mapping for rdkit object classes
bond_stereo_mappings = {Chem.rdchem.BondStereo.STEREOANY: 'STEREOANY',
                        Chem.rdchem.BondStereo.STEREOCIS: 'STEREOCIS',
                        Chem.rdchem.BondStereo.STEREOE: 'STEREOE',
                        Chem.rdchem.BondStereo.STEREONONE: 'STEREONONE',
                        Chem.rdchem.BondStereo.STEREOTRANS: 'STEREOTRANS',
                        Chem.rdchem.BondStereo.STEREOZ: 'STEREOZ'}

chiral_tag_mappings = {Chem.rdchem.ChiralType.CHI_OTHER: 'CHI_OTHER',
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 'CHI_TETRAHEDRAL_CCW',
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 'CHI_TETRAHEDRAL_CW',
                       Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 'CHI_UNSPECIFIED'}

bond_type_mappings = {'CFG=1': 'up',
                      'CFG=2': 'either',
                      'CFG=3': 'down'}


def get_svg_instances(mol: Chem.Mol,
                      svg: str,
                      bbox_params: Union[dict, None] = None,
                      split_bonds_atoms: bool = False) -> Union[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Get SMARTS intances from svg object. Calculate positions, bounding boxes, SMARTS,
    chiral and stereo information. Instances can be split into bonds and atoms.
    :param mol: RDKit molecule. [rdkit.Chem.rdchem.Mol]
    :param svg: SVG string. [str]
    :param bbox_params: Margin parameters for bounding boxes. [dict]
    :param split_bonds_atoms: Wheter to explit instances into bonds and atoms. [bool]
    :return: SMARTS instances. Union[tuple[pandas.DataFrame, pandas.Dataframe], [pandas.DataFrame]]
    """

    # parse svg file
    doc = minidom.parseString(svg)

    # get only bond and atom paths, exlucding selectors
    instances = [instance for instance in doc.getElementsByTagName('path') if 'selector' not in instance.getAttribute('class')]

    # eliminate empty instances
    instances = [instance for instance in instances if instance.getAttribute('class') != '']

    # create dictionaries to hold data
    instance_boxes = []
    for instance in instances:
        # get instance class (bond or atom)
        instance_class = instance.getAttribute('class')

        # get bond coordinates from atrribute d and transform in to numpy array
        instance_coord = re.findall("\d*\.?\d", instance.getAttribute('d'))
        instance_coord = np.array(instance_coord).astype(float)

        # reshape found coordinates skipping one element at the time (order=> F)
        instance_coord = instance_coord.reshape((2, -1), order='F').round().astype(int)

        # find max and min of X and Y axis and concatenate
        instance_coord = np.hstack([instance_coord.min(axis=1), instance_coord.max(axis=1)]).tolist()

        # append to instance boxes
        instance_boxes.append([instance_class, *instance_coord])

    # create dataframe for instance boxes
    instance_boxes = pd.DataFrame(instance_boxes, columns=['instance', 'x_min', 'y_min', 'x_max', 'y_max'])

    # group instances to get max and min coordinates since double/triple bonds might have many paths
    grouped_instances = instance_boxes.groupby('instance')
    boxes = grouped_instances[['x_min', 'y_min']].min()
    boxes[['x_max', 'y_max']] = grouped_instances[['x_max', 'y_max']].max()
    boxes.reset_index(inplace=True)

    # add instance type column
    boxes['instance_type'] = boxes.reset_index().instance.apply(lambda x: 'bond' if 'bond' in x else 'atom')

    # create margins for bounding boxes
    if not bbox_params:
        bbox_params = {'bbox_margin': 15, 'non_letter_carbon_margin': 30}

    boxes[['x_min', 'y_min']] -= bbox_params['bbox_margin']
    boxes[['x_max', 'y_max']] += bbox_params['bbox_margin']

    # add atom instances without letters
    atoms_pos = pd.DataFrame([[f'atom-{idx}',
                               round(float(path.getAttribute('cx')) - bbox_params['non_letter_carbon_margin']),
                               round(float(path.getAttribute('cy')) - bbox_params['non_letter_carbon_margin']),
                               round(float(path.getAttribute('cx')) + bbox_params['non_letter_carbon_margin']),
                               round(float(path.getAttribute('cy')) + bbox_params['non_letter_carbon_margin']),
                               'atom'] for idx, path in enumerate(doc.getElementsByTagName('circle'))],
                             columns=boxes.columns)

    # keep only atoms without letters
    atoms_pos = atoms_pos[~atoms_pos.instance.isin(instance_boxes.instance)]
    boxes = pd.concat([boxes, atoms_pos]).reset_index(drop=True)

    # get bond connectivity and stereo types
    bond_stereo_types = [bond_stereo_mappings[bond.GetStereo()] for bond in mol.GetBonds()]

    # get atoms information
    atom_chiral_tags = [chiral_tag_mappings[atom.GetChiralTag()] for atom in mol.GetAtoms()]

    # add atom and bond smiles to the boxes dataframe. Also, add stereo and chiral information
    instance_smiles = {f'atom-{idx}': path.getAttribute('atom-smiles') for idx, path in enumerate(doc.getElementsByTagName('rdkit:atom'))}
    atom_chiral_tags = {k: chiral_tag for k, chiral_tag in zip(instance_smiles.keys(), atom_chiral_tags)}
    bond_stereo_maps = {}
    for idx, (path, stereo) in enumerate(zip(doc.getElementsByTagName('rdkit:bond'), bond_stereo_types)):
        begin_atom = int(path.getAttribute("begin-atom-idx")) - 1
        end_atom = int(path.getAttribute("end-atom-idx")) - 1
        bond_key = f'bond-{idx} atom-{begin_atom} atom-{end_atom}'
        instance_smiles[bond_key] = path.getAttribute('bond-smiles')
        bond_stereo_maps[bond_key] = stereo

    boxes['instance_smarts'] = boxes.instance.map(instance_smiles)
    boxes['atom_chiral'] = boxes.instance.map(atom_chiral_tags)
    boxes['bond_stereo'] = boxes.instance.map(bond_stereo_maps)
    boxes = boxes.sort_values('instance').reset_index(drop=True)

    if split_bonds_atoms:
        atom_boxes = boxes.loc[boxes.instance_type == 'atom', [c for c in boxes.columns if c != 'bond_stereo']]
        bond_boxes = boxes.loc[boxes.instance_type == 'bond', [c for c in boxes.columns if c != 'atom_chiral']]

        # add connectivity to bonds
        bond_boxes['connecting_nodes'] = bond_boxes.instance.apply(lambda x: (int(x.split(' ')[1].split('-')[-1]),
                                                                              int(x.split(' ')[2].split('-')[-1])))
        return atom_boxes, bond_boxes
    return boxes


def convert_mol_to_svg(mol: Chem.Mol, svg_size: tuple[int, int] = (768, 768)) -> (Chem.Mol, str):
    """
    Convert RDKit molecule to SVG. Also, adds explicit hydrogens.
    :param mol: RDKit molecule. [rdkit.Chem.rdchem.Mol]
    :param svg_size: SVG size. [tuple[int, int]]]
    :return: mol with explicit hydrogens. [rdkit.Chem.rdchem.Mol], svg string. [str]
    """
    mol = rdmolops.AddHs(mol, explicitOnly=True)

    dm = Draw.PrepareMolForDrawing(mol, forceCoords=False)
    d2d = Draw.MolDraw2DSVG(svg_size[0], svg_size[1])
    d2d.drawOptions().additionalAtomLabelPadding = 0.1
    d2d.DrawMolecule(dm)
    d2d.TagAtoms(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return mol, svg


def get_mol_metadata(smiles: str) -> dict:
    """
    Generate molecule metadata from SMILES string. This metainformation is used for filtering molecules.
    :param smiles: SMILES string. [str]
    :return: Molecule metadata. [dict]
    """
    # create molecule from SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # in case SMILES error, return empty dictionary
    if not mol:
        return {}
    mol_svg, svg = convert_mol_to_svg(mol)

    # get SMARTS intances from svg object
    instances_data = get_svg_instances(mol_svg, svg, split_bonds_atoms=False)

    # get atom information
    atoms = mol.GetAtoms()
    symbols = [atom.GetSymbol() for atom in atoms]
    formal_charges = [atom.GetFormalCharge() for atom in atoms]
    aromaticities = [atom.GetIsAromatic() for atom in atoms]
    isotopes = [atom.GetIsotope() for atom in atoms]
    explicit_hs = [atom.GetNumExplicitHs() for atom in atoms]
    chiralities = [atom.GetChiralTag().name for atom in atoms]

    # image metadata from mol file
    image_metadata = {'smiles': smiles,
                      'canonical_smiles': Chem.CanonSmiles(smiles, useChiral=True),
                      'n_atoms': mol.GetNumAtoms(),
                      'n_bonds': mol.GetNumBonds(),
                      'unique_elements': dict(Counter(symbols)),
                      'unique_smarts': instances_data.instance_smarts.value_counts().to_dict(),
                      'unique_bond_stereo': instances_data[instances_data['instance_type'] == 'bond'].bond_stereo.value_counts().to_dict(),
                      'atom_chiralities': dict(Counter(chiralities)),
                      'atom_charges': dict(Counter(formal_charges)),
                      'atom_aromaticities': dict(Counter(aromaticities)),
                      'isotopes': dict(Counter(isotopes)),
                      'atom_explicit_hs': dict(Counter(explicit_hs))}
    return image_metadata


def plot_annotations_from_coco(image_data, instances_data):
    """
    Plot annotations from COCO format.
    :param instances_data: Coco-style instances data. [pd.DataFrame]
    :param image_data: Image data. [pd.DataFrame]
    :return: None
    """
    # Create figure and axes
    fig, ax = plt.subplots()

    # save svg as png in temp
    img = Image.open(image_data['file_name'].values[0])

    # Display the image
    ax.imshow(img)

    for row_idx, row in instances_data.iterrows():
        # Create a Rectangle patch
        x_min, y_min, width, height = row.bbox
        rect = patches.Rectangle((x_min, y_min),
                                 width,
                                 height,
                                 linewidth=1,
                                 edgecolor='b',
                                 facecolor='none')
        plt.text(x_min + width, y_min + height, row.instance_smarts)

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def plot_instances(instances, svg):
    """
    Plot instances from SVG str.
    :param instances: Instances data. [pd.DataFrame]
    :param svg: SVG string. [str]
    :return: None.
    """
    # Create figure and axes
    fig, ax = plt.subplots()

    # save svg as png in temp
    img = svg2png(svg)
    img = Image.open(BytesIO(img))

    # Display the image
    ax.imshow(img)

    for row_idx, row in instances.iterrows():
        # Create a Rectangle patch
        if row.instance_type == 'atom':
            color = 'r' if row.atom_chiral != 'CHI_UNSPECIFIED' else 'b'
            plt.text(row.x_max, row.y_max, row.instance)
        else:
            color = 'r' if row.bond_stereo != 'STEREONONE' else 'b'
            color = color if len(row.instance_smarts) == 1 else 'g'
        rect = patches.Rectangle((row.x_min, row.y_min),
                                 row.x_max - row.x_min,
                                 row.y_max - row.y_min,
                                 linewidth=1,
                                 edgecolor=color,
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
