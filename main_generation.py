import json
import pickle
from pathlib import Path

import hydra
import numpy as np
from indigo import Indigo
from indigo.renderer import IndigoRenderer
from omegaconf import DictConfig
from pqdm.processes import pqdm
from rdkit import Chem
from termcolor import cprint

from generation.data_generation import Augmentations, generate_sample, transform_annotations_to_coco_style
from sampling.data_sampling import PubchemData

indigo_engine = Indigo()
rend = IndigoRenderer(indigo_engine)


@hydra.main(config_path="conf", config_name="config_example", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # load data and create target filepath if it doesn't exist
    pubchem_data = PubchemData(cfg.sampling)
    pubchem_data.load_pubchem_dataset()
    pubchem_data.filter_molecules_by_sequence(cfg.sampling.sequence_filters)
    pubchem_data.generate_metadata()
    pubchem_data.filter_molecules_by_metadata(cfg.sampling.metadata_filters)

    # cap at max number of molecules
    if pubchem_data.data.shape[0] > cfg.sampling.max_number_data:
        cprint(f'Generated molecules capped at {cfg.sampling.max_number_data}', 'green')
        pubchem_data.data = pubchem_data.data.sample(cfg.sampling.max_number_data, random_state=cfg.global_seed)

    # create target filepath if it doesn't exist
    Path(cfg.generation.generated_data_folder).mkdir(parents=True, exist_ok=True)
    print(f'Total number of loaded molecules in sampled PubChem data: {pubchem_data.data.shape[0]}')

    # get rdkit reconstructed smiles
    pubchem_data.data['reconstructed_smiles'] = pubchem_data.data.smiles.apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))

    # get augmentations
    augmenter = Augmentations(img_size=cfg.generation.img_size,
                              pseudo_prob=cfg.generation.pseudo_prob,
                              pseudo_type=cfg.generation.pseudo_type,
                              pseudoatoms_lib_path=cfg.generation.pseudoatoms_lib_path,
                              aug_sample_params=cfg.generation.aug_sample_params,
                              seed=cfg.global_seed)
    pubchem_data.data['augmentations'] = [augmenter.sample_augmentation() for _ in range(pubchem_data.data.shape[0])]

    # inputs for parallelization
    bbox_margins = np.random.randint(*cfg.generation.bbox_margin, (pubchem_data.data.shape[0],))
    non_letter_carbon_margins = np.random.randint(*cfg.generation.non_letter_carbon_margin, (pubchem_data.data.shape[0],))
    pubchem_data.data['bbox_params'] = [{'bbox_margin': bbox_margin,
                                         'non_letter_carbon_margin': non_carbon_margin} for bbox_margin, non_carbon_margin in zip(bbox_margins,
                                                                                                                                  non_letter_carbon_margins)]

    indigo_engine.setOption("render-image-height", cfg.generation.img_size)
    indigo_engine.setOption("render-image-width", cfg.generation.img_size)

    pqdm_args = pubchem_data.data[['cid', 'reconstructed_smiles', 'smiles', 'augmentations', 'bbox_params']].to_dict(orient='records')
    results = pqdm(pqdm_args,
                   generate_sample,
                   n_jobs=cfg.generation.n_jobs,
                   argument_type='kwargs',
                   desc='Generating image samples with bbox annotations.')

    # filter mols with error
    results = [r for r in results if isinstance(r, dict)]
    pickle.dump(results, open(f'{cfg.root_folder}/data/generated_data_temp.pkl', 'wb'))
    cprint(f'Molecules without indigo problems: {len(results)}', 'green')

    # generate coco-styled annotations
    coco_dataset = transform_annotations_to_coco_style(results, cfg.coco_dataset_metadata)

    # save synthetic data annotations
    json.dump(coco_dataset, open(f'{cfg.root_folder}/data/{cfg.annotations_json_filename}', 'w'))


if __name__ == '__main__':
    main()
