# ``NS-Slicer``: A Learning-Based Approach to Static Program Slicing
Replication package for our paper, conditionally accepted at OOPSLA'24.

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup 

#### 1. Project Environment
Currently, ``NS-Slicer`` works flawlessly on Linux, and can be set up easily with all the prerequisite packages by following these instructions:
  1. Download appropriate version of [conda](https://repo.anaconda.com/miniconda/) for your machine.
  2. Install  it by running the `conda_install.sh` file, with the command:
     ```bash
     $ bash conda_install.sh
     ```
  3. Add `conda` to bash profile:
     ```bash
     $ source ~/.bashrc
     ```
  4. Navigate to ``ns-slicer`` (top-level directory) and create a conda virtual environment with the included `environment.yml` file using the following command:
     
     ```bash
     $ conda env create -f environment.yml
     ```
  5. Activate the virtual environment with the following command:
     
     ```bash
     $ conda activate autoslicer
     ```

#### 2. Data Artifacts
Navigate to ``ns-slicer/data/`` to find the dataset files (``{train|val|test}-examples.json``) -- use these files to benchmark learning-based static slicing approaches, or replicate results from the paper.

#### 3. Model Artifacts
Navigate to ``ns-slicer/models/`` to find the trained model weights with CodeBERT and GraphCodeBERT pre-trained language models -- use these files to replicate results from the paper, or to produce static program slices.

### Usage
See [link](https://github.com/aashishyadavally/ns-slicer/tree/main/src/README.md) for more details.

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements, to ensure that `ns-slicer`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/aashishyadavally/storyteller/blob/master/LICENSE) file for more details.
