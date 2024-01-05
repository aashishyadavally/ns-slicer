# ``NS-Slicer``: A Learning-Based Approach to Static Program Slicing
Replication package for our paper, conditionally accepted at OOPSLA'24.

## Contents

* [Getting Started](#getting-started)
  - [Setup](setup)
    - [Hardware Requirements](#hardware-requirements)
    - [Project Environment](#project-environment)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage)
* [Contributing Guidelines](#contributing-guidelines)
* [License](#license)

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup 

#### Hardware Requirements
``NS-Slicer`` requires a GPU to run *fast* and produce the results. On machines without a GPU, note that it can be notoriously slow.

#### Project Environment
Currently, ``NS-Slicer`` works well on Ubuntu OS, and can be set up easily with all the prerequisite packages by following these instructions (skip steps 1 - 3 if ``conda`` is already installed):
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

### Directory Structure

#### 1. Data Artifacts
Navigate to ``ns-slicer/data/`` to find the dataset files (``{train|val|test}-examples.json``) -- use these files to benchmark learning-based static slicing approaches, or replicate results from the paper.

#### 2. Model Artifacts
Navigate to ``ns-slicer/models/`` to find the trained model weights with CodeBERT and GraphCodeBERT pre-trained language models -- use these files to replicate results from the paper, or to produce static program slices for custom Java programs.

#### 3. Code
Navigate to ``ns-slicer/src/`` to find the source code for running experiments/using NS-Slicer to predict backward and forward static slices for a Java program.

#### 4. Preliminary Study
Navigate to ``ns-slicer/empirical-study/`` to find the details from the preliminary empirical study (see Section 3) in the paper.

### Usage
See [link](https://github.com/aashishyadavally/ns-slicer/tree/main/src/README.md) for details about replicating results in the paper, as well as using ``NS-Slicer`` to predict static program slices for Java programs.

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* Code should carry appropriate comments, wherever necessary, and follow the docstring convention in the repository.

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements, to ensure that `ns-slicer`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/aashishyadavally/ns-slicer/tree/main/LICENSE) file for more details.
