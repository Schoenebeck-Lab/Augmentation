# AIxChem
The AIxChem framework provides a collection of machine learning tools to be used for exploration of chemical space with minimal data.

## Getting started
In order to use the framework, ensure you have all the required dependencies installed. The required environment files can be found in the requirements/ directory. To create and activate an environment follow the instructions provided below:

### Installation
Prepare your anaconda environment. Upon first installation, it is recommended to create a new environment.
```sh
conda env create --file environment.yml --name myenv
conda activate myenv
```

If you are updating from an older version of aixchem, you can just update the environment.
```sh
conda activate myenv
conda env update --file environment.yml
```

Install package to your current virtual environment using pip.
```sh
pip install .
```

All the information regarding the different functions of the pipeline can be found in the Wiki of this repository: https://github.com/Schoenebeck-Lab/Augmentation/wiki
