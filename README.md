# Evaluation of Training Strategies with Multi-Expert Consensus for Classification and Precise Localisation of Barrett's Neoplasia
 
 This repository contains the codebases for the following publication:
 - Carolus H.J. Kusters *et al.* - Evaluation of Training Strategies with Multi-Expert Consensus for Classification and Precise Localisation of Barrett's Neoplasia <br />  *Submission under Review*

## Folder Structure
The folder structure of this repository is as follows:

```bash
├── data
│   └── dataset.py
├── models
│   ├── UNet.py
│   └── model.py
├── preprocess
│   └── generate_cache.py
├── utils
│   ├── loss.py
│   ├── metrics.py
│   └── optim.py
├── conda.yaml
├── train.sh
├── inference.py
├── train.py
└── README.md
```
 
## Environment
The required docker image for a SLURM cluster with apptainer is specified in "train.sh".
The conda environment with the software dependencies for this project are specified in "conda.yaml".


## Citation

If you think this helps, please use the following citation for the conference paper:
```bash
...
```