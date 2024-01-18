# Artifact for "A Learning-Based Approach to Static Program Slicing"

NS-Slicer is a learning-based static program slicing tool, which extends such an analysis to partial Java programs. The source code, data, and model artifacts are publicly available on GitHub (https://github.com/aashishyadavally/ns-slicer), and Zenodo (https://zenodo.org/records/10463878).

## Purpose
**Submission for OOPSLA 2024 Artifact:**
* **Available Badge:** We provide the artifact with a permanent DOI from Zenodo and also maintain a public GitHub repository for the project.
* **Reusable Badge:** We describe how to reproduce the major results in the paper, and use the tool to slice Java programs.


## Table of Contents

* [Getting Started](#getting-started)
  - [Setup](#setup)
    - [Hardware Requirements](#hardware-requirements)
    - [Project Environment](#project-environment)
  - [Directory Structure](#directory-structure)
  - [Usage Guide](#usage-guide)
* [Contributing Guidelines](#contributing-guidelines)
* [License](#license)

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup 

#### Hardware Requirements
``NS-Slicer`` requires a GPU to run *fast* and produce the results. On machines without a GPU, note that it can be notoriously slow.

#### Project Environment
Currently, ``NS-Slicer`` works well on Ubuntu OS, and can be set up easily with all the prerequisite packages by following these instructions (if ``conda`` is already installed, update to the latest version with ``conda update conda``, and skip steps 1 - 3):
  1. Download the latest, appropriate version of [conda](https://repo.anaconda.com/miniconda/) for your machine (tested with ``conda 23.11.0``).
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

     To test successful installation, make sure ``autoslicer`` appears in the list of conda environments returned with ``conda env list``.
  5. Activate the virtual environment with the following command:
     
     ```bash
     $ conda activate autoslicer
     ```

### Directory Structure

#### 1. Data Artifacts
Navigate to ``ns-slicer/data/`` to find:
* the dataset files (``{train|val|test}-examples.json``) -- use these files to benchmark learning-based static slicing approaches, or replicate intrinsic evaluation results in the paper (Sections 6.1 - 6.3).
* aliasing dataset files (``aliasing-{examples|dataloader}.pkl``) -- use these files to replicate variable aliasing experiment in the paper (Section 6.4).
* vulnerability detection dataset file (``filtered-methods.json``) -- use this file to replicate extrinsic evaluation experiment in the paper (Section 6.5).

#### 2. Model Artifacts
Navigate to ``ns-slicer/models/`` to find the trained model weights with CodeBERT and GraphCodeBERT pre-trained language models -- use these files to replicate results from the paper, or to produce static program slices for custom Java programs.

#### 3. Code
Navigate to ``ns-slicer/src/`` to find the source code for running experiments/using NS-Slicer to predict backward and forward static slices for a Java program.

#### 4. Preliminary Study
Navigate to ``ns-slicer/empirical-study/`` to find the details from the preliminary empirical study (see Section 3) in the paper.

### Usage Guide
See [link](https://github.com/aashishyadavally/ns-slicer/tree/main/src/README.md) for details about replicating results in the paper, as well as using ``NS-Slicer`` to predict static program slices for Java programs. Here's an executive summary of the same:

| Experiment                                        | Table # in Paper | Data Artifact(s)                             | Run Command(s)                                                        | Model Artifact(s) for Direct Inference |
| ---                                               | :----:           | :---:                                        | :---:                                                                 | :---:                                  |
| **(RQ1)** Intrinsic Evaluation on *Complete Code* | 1                | ``data/{train\|val\|test}-examples.json``    |  [click here](src/README.md/#intrinsic-evaluation-on-complete-code)   | [CodeBERT, rows 7-9](https://drive.google.com/drive/folders/1wxyL6pRESee4WSFMuX0EmCEsRYlxSwvD?usp=share_link)                 |
|                                                   |                  |                                              |                                                                       | [GraphCodeBERT, rows 10-12](https://drive.google.com/drive/folders/1zq0NUt7WFXfu4Q5b3oLrHLq_iffv-r5M?usp=share_link)          |
| **(RQ2)** Intrinsic Evaluation on *Partial Code*  | 2                | ``data/{train\|val\|test}-examples.json``    |  [click here](src/README.md/#intrinsic-evaluation-on-partial-code)    | [GraphCodeBERT](https://drive.google.com/drive/folders/1zq0NUt7WFXfu4Q5b3oLrHLq_iffv-r5M?usp=share_link)                      |
| **(RQ3)** Ablation Study                          | 3                | ``data/{train\|val\|test}-examples.json``    |  [click here](src/README.md/#ablation-study)                          | -                                      |
| **(RQ4)** Variable Aliasing                       | 4                | ``data/aliasing-{examples\|dataloader}.pkl`` |  [click here](src/README.md/#variable-aliasing/)                      | [CodeBERT, rows 1-2](https://drive.google.com/drive/folders/1wxyL6pRESee4WSFMuX0EmCEsRYlxSwvD?usp=share_link)                |
|                                                   |                  |                                              |                                                                       | [GraphCodeBERT, rows 3-4](https://drive.google.com/drive/folders/1zq0NUt7WFXfu4Q5b3oLrHLq_iffv-r5M?usp=share_link)           |
| **(RQ5)** Extrinsic Evaluation                    | 5                | ``data/filtered-methods.json``               |  [click here](src/README.md/#extrinsic-evaluation/)                   | [GraphCodeBERT, row 2](https://drive.google.com/drive/folders/1zq0NUt7WFXfu4Q5b3oLrHLq_iffv-r5M?usp=share_link)   |

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* Code should carry appropriate comments, wherever necessary, and follow the docstring convention in the repository.

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements, to ensure that `ns-slicer`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/aashishyadavally/ns-slicer/tree/main/LICENSE) file for more details.
