<div align="center">
<h3>Evaluation of Training Strategies with Multi-Expert Consensus for Classification and Precise Localisation of Barrett's Neoplasia</h3>

[Carolus H.J. Kusters](https://github.com/chjkusters)<sup>1 :email:</sup>, Tim G.W. Boers<sup>1</sup>, Tim J.M. Jaspers<sup>1</sup>, Martijn R. Jong<sup>2</sup>, Rixta A.H. van Eijck van Heslinga<sup>2</sup>, Albert J. de Groof<sup>2</sup>, Jacques J. Bergman<sup>2</sup>, Fons van der Sommen<sup>1</sup>, Peter H.N. de With<sup>1</sup>
 
<sup>1</sup>  Department of Electrical Engineering, Video Coding & Architectures, Eindhoven University of Technology <br /> <sup>2</sup>  Department of Gastroenterology and Hepatology, Amsterdam University Medical Centers, University of Amsterdam

(<sup>:email:</sup>) corresponding author

*Third Workshop on Cancer Prevention, detection, and intervenTion &#40;CaPTion&#41; - Satellite Event MICCAI 2024* <br /> ([Proceeding](...))

</div>


[//]: # ( This repository contains the codebases for the following publication:)

[//]: # ( - Carolus H.J. Kusters *et al.* - Evaluation of Training Strategies with Multi-Expert Consensus for Classification and Precise Localisation of Barrett's Neoplasia <br />  *Third Workshop on Cancer Prevention, detection, and intervenTion &#40;CaPTion&#41; - Satellite Event MICCAI 2024*)

## Abstract
Place Abstract here...


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