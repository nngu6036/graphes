# GraphER: Generating Realistic Graphs from Degree of Sequqnces with Edge Rerewiring
This repository is the official PyTorch implementation of GraphDS, a graph generative model using auto-regressive model.

## Installation
Install PyTorch following the instuctions on the [official website](https://pytorch.org/). The code has been tested over PyTorch 0.2.0 and 0.4.0 versions.
```bash
conda install pytorch torchvision cuda90 -c pytorch
```
Then install the other dependencies.
```bash
pip install -r requirements.txt
```

## Generate input dataset
```bash
python create_dataset.py  --dataset-prefix dataset1 --config  dataset1_config.toml
```

## Train and evaluate MS-VAE model
```bash
python train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model msvae_community --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_qm9_edgelists --config  msvae_config1.toml --output-model msvae_qm9 --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_zinc_edgelists --config  msvae_config1.toml --output-model msvae_zinc --evaluate
```

## Evaluate MS-VAE model
```bash
python train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --input-model msvae_ego --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --input-model msvae_community --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_qm9_edgelists --config  msvae_config1.toml --input-model msvae_qm9 --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_zinc_edgelists --config  msvae_config1.toml --input-model msvae_zinc --evaluate
```

## Train and evaluate Std-VAE model
```bash
python train_stdvae.py  --dataset-dir dataset1_planar_edgelists --config  msvae_config1.toml --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml  --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_sbm_edgelists --config  msvae_config1.toml  --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml  --evaluate
```

## Training and evaluate GraphER model
```bash
python train_grapher.py --config grapher_config.toml  --dataset-dir dataset1_planar_edgelists --msvae-model msvae_planar --msvae-config msvae_config1.toml --output-model grapher_planar --evaluate
```
```bash
python train_grapher.py --config grapher_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate 
```
```bash
python train_grapher.py --config grapher_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate 
```
## Evaluate GraphER model
```bash
python train_grapher.py --config grapher_config.toml  --dataset-dir dataset1_planar_edgelists --msvae-model msvae_planar --msvae-config msvae_config1.toml --input-model grapher_planar --evaluate
```
```bash
python train_grapher.py --config grapher_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate 
```
