# Dataset sources for the camera-ready evaluation

This inventory matches the 29 distinct task names in `results/paper14_crossdomain.csv`; the MV, VTAB, and EX tasks are subsets after normalizing `pets` to `oxford_pets`. It is an attribution map, not proof of historical cache provenance or a complete license audit. Citations appear in both papers' Appendices B–E and are defined in `paper/references.bib`.

| Result-table task | Original publication / release citation key(s) | Source |
|---|---|---|
| `cifar10` | `krizhevsky2009cifar` | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| `cifar100` | `krizhevsky2009cifar` | Same original CIFAR report |
| `cifar100_coarse` | `krizhevsky2009cifar` | CIFAR-100 superclass labels, not a separately collected dataset |
| `dtd` | `cimpoi2014dtd` | [Describing Textures in the Wild](https://www.robots.ox.ac.uk/~vgg/publications/2014/Cimpoi14/cimpoi14.pdf) |
| `eurosat` | `helber2017eurosat` | [EuroSAT](https://arxiv.org/abs/1709.00029) |
| `fashion_mnist` | `xiao2017fashion` | [Fashion-MNIST](https://arxiv.org/abs/1708.07747) |
| `fgvc_aircraft` | `maji2013aircraft` | [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) |
| `flowers102` | `nilsback2008flowers` | [Flowers-102](https://www.robots.ox.ac.uk/~vgg/research/flowers/) |
| `food101` | `bossard2014food` | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |
| `oxford_pets` | `parkhi2012pets` | [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/); alias `pets` |
| `stl10` | `coates2011stl10` | [STL-10](https://cs.stanford.edu/~acoates/stl10/) |
| `svhn` | `netzer2011svhn` | [SVHN](https://ufldl.stanford.edu/housenumbers/) |
| `mnist` | `lecun1998mnist` | [MNIST](https://yann.lecun.com/exdb/mnist/) |
| `usps` | `hull1994usps` | [Hull's original database paper](https://doi.org/10.1109/34.291440) |
| `gtsrb` | `stallkamp2011gtsrb` | [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) |
| `semeion` | `semeion1998digits` | [Original dataset record](https://doi.org/10.24432/C5SC8V); use its 1998 citation date, not the later UCI donation date |
| `beans` | `makererebeans` | [Makerere AI Lab original iBean release](https://github.com/AI-Lab-Makerere/ibean); no unverified standalone paper assigned |
| `rendered_sst2` | `socher2013sst`, `radford2021clip` | [Original SST](https://aclanthology.org/D13-1170/) and [CLIP rendered variant](https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md) |
| `pathmnist` | `yang2023medmnist`, `kather2019path` | [MedMNIST](https://medmnist.com/v2), colorectal histology source |
| `dermamnist` | `yang2023medmnist`, `tschandl2018ham`, `codella2019isic` | MedMNIST; HAM10000 / ISIC |
| `octmnist` | `yang2023medmnist`, `kermany2018medical` | MedMNIST; Kermany et al. retinal OCT images |
| `pneumoniamnist` | `yang2023medmnist`, `kermany2018medical` | MedMNIST; Kermany et al. chest X-rays |
| `retinamnist` | `yang2023medmnist`, `liu2022deepdrid` | MedMNIST; [DeepDRiD](https://github.com/deepdrdoc/DeepDRiD), not EyePACS |
| `breastmnist` | `yang2023medmnist`, `aldhabyani2020breast` | MedMNIST; Al-Dhabyani et al. breast ultrasound |
| `bloodmnist` | `yang2023medmnist`, `acevedo2020blood` | MedMNIST; Acevedo et al. peripheral blood images |
| `tissuemnist` | `yang2023medmnist`, `ljosa2012tissue` | MedMNIST; BBBC kidney-cortex microscopy images |
| `organamnist` | `yang2023medmnist`, `bilic2019lits`, `xu2019organs` | MedMNIST; LiTS images plus organ annotations, axial view |
| `organcmnist` | `yang2023medmnist`, `bilic2019lits`, `xu2019organs` | Same sources, coronal view |
| `organsmnist` | `yang2023medmnist`, `bilic2019lits`, `xu2019organs` | Same sources, sagittal view |

The MedMNIST source-paper mapping follows its [official source bibliography](https://medmnist.com/assets/v2/medmnist.bib); RetinaMNIST additionally cites the [DeepDRiD authors' publication record](https://github.com/deepdrdoc/DeepDRiD). These biomedical tasks are standardized derivatives, not new images collected in this work. ChestMNIST, 3D MedMNIST tasks, and KMNIST are not present in the evaluated 29-task table and are not added to the claimed evaluation inventory.

ImageNet is separately credited by `deng2009imagenet` ([original project](https://www.image-net.org/)). It is used for pretraining and appears in the historical SSL reference-script configuration; it is not one of the headline target tasks. The original VTAB benchmark is credited by `zhai2019large`; our retained “VTAB pool” is not the complete original suite.

STL-10 occurs in cached diagnostic inventories, but is excluded from all methods' 22-unit target-selection aggregate. References are target-disjoint in that aggregate; this inventory does not repair missing historical SSL reference manifests.
