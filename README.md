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
python .\main_dataset.py --config-path dataset1_config.toml --output-prefix dataset1
```

## Training MS-VAE model
```bash
python .\main_dataset.py --config-path dataset1_config.toml --output-prefix dataset1
```

## Training GraphES model
```bash
python .\main_dataset.py --config-path dataset1_config.toml --output-prefix dataset1
```


