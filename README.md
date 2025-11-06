Bidirectional-Importance Goal-Driven Pruning (BGDP)
==================================================

This repository provides the code to reproduce the experiments in
“A Bidirectional Importance Metric for Goal-Driven Pruning” (Akpolat, 2025).


--------------------------------------------------
1) Code & Data Availability
--------------------------------------------------
- Code: this repository (three scripts under src/).
- Pretrained checkpoints Zenodo DOI: https://doi.org/10.5281/zenodo.17338069

Where to put the checkpoints?
- Place the downloaded .h5 files in the same folder as the scripts:
  - src/MLP_Model.h5
  - src/Modified_VGG16_CIFAR10.h5
- (ImageNet/VGG16 uses Keras Applications and needs no local checkpoint.)

--------------------------------------------------
2) Environment
--------------------------------------------------
Tested with Python 3.10 and the following packages:
- tensorflow>=2.10,<2.13
- keras>=2.9,<2.13
- kerassurgeon>=0.2.0
- numpy>=1.23
- matplotlib>=3.6
- h5py>=3.7
- scikit-learn>=1.2
- tqdm>=4.65

Create with conda (recommended):
  conda env create -f environment.yml
  conda activate bgdp-env

Or with pip:
  python -m venv .venv
  (Windows) .venv\Scripts\activate
  (Linux/Mac) source .venv/bin/activate
  pip install -r requirements.txt

--------------------------------------------------
3) How to Reproduce
--------------------------------------------------
Change into the scripts folder:
  cd src

a) MNIST (MLP, Dense) — expects src/MLP_Model.h5:
  python pruning_mlp_mnist.py

b) CIFAR-10 (Modified VGG16) — expects src/Modified_VGG16_CIFAR10.h5:
  python pruning_modified_vgg16_cifar10.py

c) ImageNet (VGG16 via Keras Applications) — no local checkpoint required:
  python pruning_vgg16_imagenet.py

Note: Pruned model files are written in the same folder as the script by default
(e.g., src/<pruned_name>.h5).
