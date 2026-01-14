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
python create_dataset.py  --dataset-dir datasets --config  config/dataset1_config.toml
```

## Plot distance 
```bash
python plot_distance.py  --dataset-dir datasets/community_edgelists --config  configs/distance_config.toml
python plot_distance.py  --dataset-dir datasets/ego_edgelists --config  config/distance_config.toml
```

## Compute PE for PyG dataset
```bash
python compute_pe.py  --dataset QM9 --dataset-dir datasets/QM9 --k 8
python compute_pe.py  --dataset ZINC --dataset-dir datasets/ZINC --k 16 --subset
```


## Train and evaluate MS-VAE model
```bash
python train_msvae.py  --dataset-dir datasets/ego_edgelists --config  config/msvae_config1.toml --output-model models/msvae_ego --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/grid_edgelists --config  config/msvae_config1.toml --output-model models/msvae_grid --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/community_edgelists  --config config/ msvae_config1.toml --output-model models/msvae_community --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/QM9 --config  config/msvae_config1.toml --output-model models/msvae_qm9 --dataset QM9 --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/ZINC --config  config/msvae_config1.toml --output-model models/msvae_zinc --dataset ZINC --evaluate
```

## Evaluate MS-VAE model
```bash
python train_msvae.py  --dataset-dir datasets/ego_edgelists --config  config/msvae_config1.toml --input-model models/msvae_ego --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/community_edgelists --config  config/msvae_config1.toml --input-model models/msvae_community --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/grid_edgelists --config  config/msvae_config1.toml --input-model models/msvae_grid --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/QM9 --config  config/msvae_config1.toml --input-model models/msvae_qm9  --dataset QM9  --evaluate
```
```bash
python train_msvae.py  --dataset-dir datasets/ZINC --config  config/msvae_config1.toml --input-model models/msvae_zinc  --dataset ZINC --evaluate
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
```bash
python train_grapher.py --config grapher_qm9_config.toml  --dataset-dir QM9 --msvae-model msvae_qm9 --msvae-config msvae_config1.toml --output-model grapher_qm9 --dataset QM9  --evaluate 
```
```bash
python train_grapher.py --config grapher_zinc_config.toml  --dataset-dir ZINC --msvae-model msvae_zinc --msvae-config msvae_config1.toml --output-model grapher_zinc --dataset ZINC  --evaluate 
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
