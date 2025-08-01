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

## Compute mixing time
```bash
python plot_mixing_time.py  --dataset-dir dataset1_ego_edgelists --config  plot_config.toml
```

## Train and evaluate MS-VAE model
```bash
python train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_grid_edgelists --config  msvae_config1.toml --output-model msvae_grid --evaluate
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
python train_msvae.py  --dataset-dir dataset1_grid_edgelists --config  msvae_config1.toml --input-model msvae_grid --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_qm9_edgelists --config  msvae_config1.toml --input-model msvae_qm9 --evaluate
```
```bash
python train_msvae.py  --dataset-dir dataset1_zinc_edgelists --config  msvae_config1.toml --input-model msvae_zinc --evaluate
```

## Train and evaluate Std-VAE model
```bash
python train_stdvae.py  --dataset-dir dataset1_grid_edgelists --config  msvae_config1.toml --output-model stdvae_grid  --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml  --output-model stdvae_ego --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model stdvae_community --evaluate
```
## Evaluate Std-VAE model
```bash
python train_stdvae.py  --dataset-dir dataset1_grid_edgelists --config  msvae_config1.toml  --input-model stdvae_grid  --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml  --input-model stdvae_ego  --evaluate
```
```bash
python train_stdvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml  --input-model stdvae_community  --evaluate
```

## Training and evaluate GraphER model
```bash
python train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate 
```
```bash
python train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate 
```
```bash
python train_grapher.py --config grapher_grid_config.toml  --dataset-dir dataset1_grid_edgelists --msvae-model msvae_grid --msvae-config msvae_config1.toml --output-model grapher_grid --evaluate 
```
## Evaluate GraphER model
```bash
python train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate 
```
```bash
python train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate 
```
```bash
python train_grapher.py --config grapher_grid_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_grid --msvae-config msvae_config1.toml --input-model grapher_grid --evaluate 
```
```bash
python train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation
```
```bash
python train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation
```
```bash
python train_grapher.py --config grapher_grid_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_grid --msvae-config msvae_config1.toml --input-model grapher_grid --evaluate  --ablation
```