<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">CEDe: A collection of expert-curated datasets with atom-level entity annotations for 
Optical Chemical Structure Recognition</h3>
</div>

<!-- ABOUT THE PROJECT -->
### About CEDe 

![CEDeMainDiagram](https://storage.googleapis.com/lgcede/CEDe%20-%20MainDiagram.png)

**O**ptical **C**hemical **S**tructure **R**ecognition (OCSR) deals with the translation from chemical images 
to molecular structures, which is the main way chemical compounds are depicted in scientific 
documents. Traditional rule-based methods follow a framework based on the detection of atoms 
and bonds, followed by the reconstruction of the compound structure. Recently, neural 
architectures analog to image captioning have been explored to solve this task, yet they 
still show to be data inefficient, using millions of examples just to show performance 
comparable with traditional methods. Looking to motivate and benchmark new approaches 
based on atomic-level entities detection and graph reconstruction, we present CEDe, 
a unique collection of chemical entity bounding boxes manually curated by experts for 
scientific literature datasets. These annotations combine to more than 700,000 chemical 
entity bounding boxes with the necessary information for structure reconstruction. Also, a 
large synthetic dataset containing 1 million molecular images and annotations is released 
in order to explore transfer-learning techniques that could help these architectures perform
better under low-data regimes. Benchmarks show that detection-reconstruction based models
can achieve performances on par with or better than image captioning-like models, 
even with 100x fewer training examples.

This repository is the contains currently contains the code for sampling, synthetic data generation and
visualization of the **CEDe** dataset. Models and implementations for benchmarks will be released soon.

## Download CEDe
We provide different options for downloading the CEDe dataset. Image data and annotations can be downloaded
separately or as one compressed file. Also, different dataset sizes are provided (every smaller dataset
is fully contained in bigger versions).

#### CEDe real data

[Full (135.7MB)](https://storage.googleapis.com/lgcede/CEDe_dataset_v0.2.tar.gz) | 
[Annotations (194MB)](https://storage.googleapis.com/lgcede/CEDe_dataset_v0.2.json) |
[Train split annotations (38.5MB)](https://storage.googleapis.com/lgcede/CEDe_dataset_finetune_split_v0.2.json) |
[Test split annotations (156MB)](https://storage.googleapis.com/lgcede/CEDe_dataset_test_split_v0.2.json) | 
[Images (53.6MB)](https://storage.googleapis.com/lgcede/CEDe_dataset_images_v0.2.tar.gz)

#### Synthetic data
**10K images**: [Full (334MB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_10k.tar.gz) | [Annotations (177MB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_10k.json) | [Images (320MB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_images_10k.tar.gz)

**50K images**: [Full (1.6GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_50k.tar.gz) | [Annotations (887MB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_50k.json) | [Images (1.6GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_images_50k.tar.gz)

**100K images**: [Full (3.3GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_100k.tar.gz) | [Annotations (1.7GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_100k.json) | [Images (3.1GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_images_100k.tar.gz)

**1M images**: [Full (32.5GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_1M.tar.gz) | [Annotations (17.3GB)](https://storage.googleapis.com/lgcede/CEDe_synthetic_data_1M.json) | [Images (31.2G)](https://storage.googleapis.com/lgcede/CEDe_synthetic_images_1M.tar.gz)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* PIP
  ```sh
  git clone https://github.com/rshormazabal/CEDe.git
  cd CEDe
  pip install -r requirements.txt
  ```
* Conda
  ```sh
  git clone https://github.com/rshormazabal/CEDe.git
  cd CEDe
  conda env create -f environment.yml -n cede_generation
  ```

<!-- USAGE EXAMPLES -->
## Config file documentation
A detailed example for the config file can be found in `./conf/config_example.yaml`.
```yaml
root_folder: Project root folder. [str] 
annotations_json_filename: Synthetic CEDe annotations JSON filename. [str] 
sampling:
  data_path: Data folder path. [str]
  dataset_name: Whether to use 'SMILES' or 'InChI' as dataset. [str](SMILES, InChI)
  download: Redownload PUBCHEM dataset. [bool]
  load_pickle: Load previously generated metadata pickle file. [bool]
  pickle_filename: Metadata pickle filename. [str]

  n_jobs: Number of jobs for metadata generation. [int]

  pubchem_nrows: Number of rows to read out of pubchem file before filtering. [int]
  max_number_data: Maximum number of data to generate after filtering. [int]

  # parameters to filter cases
  sequence_filters:
    # SMILES compounds containing these characters are kept. However, other token are also 
    # included in final annotations, since they can be contained within these structures.
    chars_to_keep: List of non-atom characters to filter dataset. [list of str]
    atoms_to_keep: List of atoms to filter dataset. [list of str]
    # SMILES compounds containing these characters will be removed.
    chars_to_drop: List of characters to remove structures from dataset. [list of str]
    max_len: Maximum length of SMILES string. [int]
    min_len: Minimum length of SMILES string. [int]

  metadata_filters:
    max_atoms: Maximum number of atoms in a molecule (not characthers). [int]

generation:
  generated_data_folder: Path to store generated data. [str] 
  pseudo_type: Sets the pseudoatom generation from ['R', 'random', 'given']. [str]
  # 'R' generates on only '[R{number}]' style pseudoatoms. 
  # 'random' generates a random string to attach as pseudoatom.
  # 'given' choses from the file specified in 'pseudoatomos_lib_path' 
  pseudo_prob: Probability to replace an atom with a pseudoatom. [float]
  img_size: Size of generated images. [int]
  n_jobs: Number of jobs for the data generation process. [int]
  pseudoatoms_lib_path: Path to pseudoatoms library csv file. [str] 
  bbox_margin: Range for letter instance margins. (int, int)         
  non_letter_carbon_margin: Range for non-letter carbon instance margins. (int, int) 
  aug_sample_params:
    linewidth_range: Range for linewidth augmentation. (int, int)
    font_size_range: Range for font size augmentation. (int, int)
    rotation_angle_range: Range for rotation angle augmentation. (int, int)
    xy_sheer_range: Range for xy sheer augmentation. (int, int)
    fonts_path: Path to library of font files. [str]
coco_dataset_metadata:
  description: Description of the dataset instance. [str] 
  url: URL to download dataset. [str] 
  version: Identifier version. [str]
  year: Year of dataset creation. [int]
  contributor: Contributor name. [str]
  creation_date: Date of dataset creation. [str]
  license_url: License URL. [str] 
  license_name: License name. [str]
global_seed: Global seed for the project (sets numpy, pandas, random, torch, etc). [int]
```

## How to run
After installing dependencies, you can generate data by specifying a config file.
For specificying specific parameters directly on the CLI, refer to [Hydra documentation](https://hydra.cc/docs/intro/).
```sh
python main_generation.py --config_file <path_to_config_file>
```
##### Example
```sh
python main_generation.py --config_file ./conf/config_example.yaml --generation.pubchem_nrows 100000 --generation.max_number_data 10000
```

## License

Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/2.0/legalcode