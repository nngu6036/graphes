# GraphDS: Generating Realistic Graphs from Degree of Sequqnces
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
python main_dataset.py  --dataset-prefix dataset1 --config-file  dataset1_config.toml
```

## Training EG-Predictor model
```bash
 python main_eg_predictor.py  --dataset-dir dataset1_planar_edgelists --output-model eg_planar_predictor

 python main_eg_predictor.py  --dataset-dir dataset1_sbm_edgelists --output-model eg_sbm_predictor
```

## Training MS-VAE model
```bash
python main_msvae.py  --dataset-dir dataset1_planar_edgelists --config-file  msvae_config1.toml --output-model msvae_planar --evaluate
python main_msvae.py  --dataset-dir dataset1_ego_edgelists --config-file  msvae_config1.toml --output-model msvae_ego --evaluate
python main_msvae.py  --dataset-dir dataset1_sbm_edgelists --config-file  msvae_config1.toml --output-model msvae_sbm --evaluate
```

## Training GraphES model
```bash
python .\main_dataset.py --config-path dataset1_config.toml --output-prefix dataset1
```


