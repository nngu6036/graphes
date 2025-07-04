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
python create_dataset.py  --dataset-prefix dataset1 --config-file  dataset1_config.toml
```

## Training MS-VAE model
```bash
python main_msvae.py  --dataset-dir dataset1_planar_edgelists --config-file  msvae_config1.toml --output-model msvae_planar --evaluate
python main_msvae.py  --dataset-dir dataset1_ego_edgelists --config-file  msvae_config1.toml --output-model msvae_ego --evaluate
python main_msvae.py  --dataset-dir dataset1_sbm_edgelists --config-file  msvae_config1.toml --output-model msvae_sbm --evaluate
python main_msvae.py  --dataset-dir dataset1_community_edgelists --config-file  msvae_config1.toml --output-model msvae_community --evaluate
```

## Training GraphER model
```bash
python main_grapher.py --config-path grapher_config.toml  --dataset-dir dataset1_planar_edgelists --msvae-model msvae_planar --msvae-config-path msvae_config1.toml --output-model grapher_planar --evaluate
```


