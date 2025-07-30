python3 train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate > /tmp/msvae_ego 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate > /tmp/graph_ego_1

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_1

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_1

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation > /tmp/graph_ego_1_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_1_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_1_a 

python3 train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate > /tmp/msvae_ego 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate > /tmp/graph_ego_2

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_2

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_2

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation > /tmp/graph_ego_2_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_2_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_2_a 

python3 train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate > /tmp/msvae_ego 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate > /tmp/graph_ego_3

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_3

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_3

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation > /tmp/graph_ego_3_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_3_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_3_a 

python3 train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate > /tmp/msvae_ego 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate > /tmp/graph_ego_4

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_4

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_4

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation > /tmp/graph_ego_4_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_4_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_4_a 

python3 train_msvae.py  --dataset-dir dataset1_ego_edgelists --config  msvae_config1.toml --output-model msvae_ego --evaluate > /tmp/msvae_ego 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --output-model grapher_ego --evaluate > /tmp/graph_ego_5 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_5 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate >> /tmp/graph_ego_5 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation > /tmp/graph_ego_5_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_5_a 

python3 train_grapher.py --config grapher_ego_config.toml  --dataset-dir dataset1_ego_edgelists --msvae-model msvae_ego --msvae-config msvae_config1.toml --input-model grapher_ego --evaluate  --ablation >> /tmp/graph_ego_5_a 