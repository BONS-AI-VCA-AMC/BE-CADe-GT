<div align="center">
<h1>Publication</h1>
<h3>Optimizing Multi-Expert Consensus for Classification and Precise Localization of Barrett's Neoplasia</h3>

[Carolus H.J. Kusters](https://chjkusters.github.io/)<sup>1 :email:</sup>, [Tim G.W. Boers](https://scholar.google.nl/citations?user=_TdckGAAAAAJ&hl=nl&oi=ao)<sup>1</sup>, [Tim J.M. Jaspers](https://scholar.google.nl/citations?user=nwfiV2wAAAAJ&hl=nl&oi=ao)<sup>1</sup>, [Martijn R. Jong](https://scholar.google.nl/citations?user=QRNrL-oAAAAJ&hl=nl&oi=ao)<sup>2</sup>, Rixta A.H. van Eijck van Heslinga<sup>2</sup>, Albert J. de Groof<sup>2</sup>, [Jacques J. Bergman](https://scholar.google.nl/citations?user=4SFBE0IAAAAJ&hl=nl&oi=ao)<sup>2</sup>, [Fons van der Sommen](https://scholar.google.nl/citations?user=qFiLkCAAAAAJ&hl=nl&oi=ao)<sup>1</sup>, Peter H.N. de With<sup>1</sup>
 
<sup>1</sup>  Department of Electrical Engineering, Video Coding & Architectures, Eindhoven University of Technology <br /> <sup>2</sup>  Department of Gastroenterology and Hepatology, Amsterdam University Medical Centers, University of Amsterdam

(<sup>:email:</sup>) corresponding author

*Third Workshop on Cancer Prevention, detection, and intervenTion &#40;CaPTion&#41; - Satellite Event MICCAI 2024* <br /> ([Proceeding](...))

</div>

## Abstract
Recognition of early neoplasia in Barrett's Esophagus (BE) is challenging, despite advances in endoscopic technology. Even with correct identification, the subtle nature of lesions leads to significant inter-observer variability in placing targeted biopsy markers and delineation of lesions. Computer-Aided Detection (CADe) systems may assist endoscopists, however, compliance of endoscopists with CADe is often suboptimal, reducing joint performance below CADe stand-alone performance. Improved localization performance of CADe could enhance compliance. These systems often use fused consensus ground-truths (GT), which may not capture subtle neoplasia gradations, affecting classification and localization. This study evaluates five consensus GT strategies from multi-expert segmentation labels and four loss functions for their impact on classification and localization performance. The dataset includes 7,995 non-dysplastic BE images (1,256 patients) and 2,947 neoplastic images (823 patients), with each neoplastic image annotated by two experts. Classification, localization for true positives, and combined detection performance are assessed and compared with 14 independent Barrett's experts. Results show that using multiple consensus GT masks with a compound Binary Cross-Entropy and Dice loss achieves the best classification sensitivity and near-expert level localization, making it the most effective training strategy. The code is made publicly available.


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