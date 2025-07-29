python3 train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model msvae_community --evaluate > /tmp/msvae_community 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate > /tmp/graph_community_1_1 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_1_2 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_1_3 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_1_1_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_1_2_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_1_3_a 

python3 train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model msvae_community --evaluate > /tmp/msvae_community 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate > /tmp/graph_community_2_1 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_2_2 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_2_3 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_2_1_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_2_2_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_2_3_a 

python3 train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model msvae_community --evaluate > /tmp/msvae_community 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate > /tmp/graph_community_3_1 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_3_2 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_3_3 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_3_1_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_3_2_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_3_3_a 

python3 train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model msvae_community --evaluate > /tmp/msvae_community 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate > /tmp/graph_community_4_1 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_4_2 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_4_3 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_4_1_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_4_2_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_4_3_a 

python3 train_msvae.py  --dataset-dir dataset1_community_edgelists --config  msvae_config1.toml --output-model msvae_community --evaluate > /tmp/msvae_community 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --output-model grapher_community --evaluate > /tmp/graph_community_5_1 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_5_2 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate > /tmp/graph_community_5_3 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_5_1_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_5_2_a 

python3 train_grapher.py --config grapher_community_config.toml  --dataset-dir dataset1_community_edgelists --msvae-model msvae_community --msvae-config msvae_config1.toml --input-model grapher_community --evaluate  --ablation > /tmp/graph_community_5_3_a 