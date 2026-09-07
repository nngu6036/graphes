# CatFlow source evidence

## S1 — uploaded VFM_CatFlow / flow_matching.py
SHA256: `c1d3ab645a62599f3ccd8b25cb90bb91fee24269842d05795e62e12f6da4593b`

Lines 27–39:
```python
  27  def conditional_velocity(distribution, x_1, t, tau_t, mu, sigma, edge=False):
  28      if distribution == 'normal':
  29          x_0 = torch.randn_like(x_1)
  30          x_0 = 0.5 * (x_0 + x_0.transpose(1, 2)) if edge else x_0
  31          diag = torch.eye(x_1.shape[1], dtype=torch.bool).unsqueeze(0).expand(x_1.shape[0], -1, -1)
  32          x_t = (1 - t) * x_0 + t * x_1
  33  
  34          v_x = x_1 - x_0
  35  
  36          noise = torch.randn_like(x_t) * 0.5
  37          noise = 0.5 * (noise + noise.transpose(1, 2)) if edge else noise
  38  
  39          x_t = x_t + noise
```

Lines 128–145:
```python
 128  def generate_graphs(model, num_mols, node_feats, edge_feats, max_nodes, device, name, mu, distribution, tau_sched, loss_function, counter, small_model):
 129      mols_per_b = num_mols // 10
 130      all_mols = []
 131  
 132      for _ in tqdm(range(10), desc='Generating molecules'):
 133          x_params, e_params = torch.zeros(mols_per_b, max_nodes, node_feats), torch.zeros(mols_per_b, max_nodes, max_nodes, edge_feats)
 134  
 135          if distribution == 'normal':
 136              x_t = torch.randn_like(x_params)
 137              e_t = torch.randn_like(e_params)
 138              e_t = (e_t + torch.transpose(e_t, 1, 2)) / 2
 139  
 140              noise_x = torch.randn_like(x_t) * 1e-6
 141              noise_e = torch.randn_like(e_t) * 1e-6
 142              noise_e = (noise_e + torch.transpose(noise_e, 1, 2)) / 2
 143  
 144              x_t = x_t + noise_x
 145              e_t = e_t + noise_e
```

Lines 581–590:
```python
 581          pred = self.neural_network(x_t, e_t, y_t, mask)
 582  
 583          if self.loss_function == 'mse':
 584              v_x, v_e = pred.X, pred.E
 585          else:
 586              x_1, e_1 = pred.X, pred.E
 587              x_1, e_1 = torch.softmax(x_1, dim=-1), torch.softmax(e_1, dim=-1)
 588  
 589              v_x = (x_1 - x_t) / (1 - t)
 590              v_e = (e_1 - e_t) / (1 - t)
```

## S2 — previous GraphER CatFlow worker, before this patch
SHA256: `608cf78928b37499b6e9d92af40d2588d013c7f4117b6ecb941c73031d508283`

Lines 79–120:
```python
  79          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
  80          ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.get("ema", .999)))
  81          history = []
  82          for epoch in range(epochs):
  83              totals = {}
  84              for split, data in (("train", train), ("val", val)):
  85                  model.train(split == "train")
  86                  indices = np.random.permutation(len(data["num_nodes"])) if split == "train" else np.arange(len(data["num_nodes"]))
  87                  total, count = 0., 0
  88                  with torch.set_grad_enabled(split == "train"):
  89                      for start in range(0, len(indices), batch_size):
  90                          ids = indices[start:start + batch_size]
  91                          x1, e1, mask, pair_mask = dense_batch(data, ids, device, dx, de)
  92                          t = torch.rand(len(ids), 1, 1, device=device)
  93                          xt, _ = flow.conditional_velocity("normal", x1, t, None, 8, 0)
  94                          et, _ = flow.conditional_velocity("normal", e1, t.unsqueeze(-1), None, 8, 0, edge=True)
  95                          # Native training zeros only the diagonal before the
  96                          # transformer; preserve this (padding is masked inside).
  97                          diag = torch.eye(et.size(1), dtype=torch.bool, device=device)[None].expand(len(ids), -1, -1)
  98                          et = et.masked_fill(diag[..., None], 0)
  99                          pred = model(xt, et, t.reshape(len(ids), 1), mask)
 100                          loss_x = F.cross_entropy(pred.X[mask], x1.argmax(-1)[mask])
 101                          loss_e = (F.cross_entropy(pred.E[pair_mask], e1.argmax(-1)[pair_mask])
 102                                    if pair_mask.any() else pred.E.sum() * 0)
 103                          loss = loss_x + 5 * loss_e
 104                          finite_loss(loss, "CatFlow loss")
 105                          if split == "train":
 106                              optimizer.zero_grad(set_to_none=True)
 107                              loss.backward()
 108                              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
 109                              optimizer.step()
 110                              ema.update(model.parameters())
 111                          total += loss.item() * len(ids)
 112                          count += len(ids)
 113                  totals[split] = total / count
 114              scheduler.step()
 115              history.append({"epoch": epoch + 1, **totals})
 116              if (epoch + 1) % int(cfg.get("log_every", 1)) == 0 or epoch + 1 == epochs:
 117                  print("CatFlow epoch %d/%d train=%.6f val=%.6f" % (epoch + 1, epochs, totals["train"], totals["val"]), flush=True)
 118          save_checkpoint(job["checkpoint"], {"model": model.state_dict(), "ema": ema.state_dict(), "ema_backend": ema_backend,
 119                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
 120                          "epoch": epochs, "history": history, "architecture": vars(args), "dx": dx, "de": de})
```

Lines 122–131:
```python
 122      else:
 123          state = torch_load(job["checkpoint"], device)
 124          model.load_state_dict(state["model"])
 125          sample = options.get("sample", {})
 126          if sample.get("use_ema", True):
 127              ExponentialMovingAverage, _ = ema_class(state.get("ema_backend", "torch_ema"))
 128              ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.get("ema", .999)))
 129              ema.load_state_dict(state["ema"])
 130              ema.to(device)
 131              ema.copy_to(model.parameters())
```

Lines 149–168:
```python
 149              x = torch.randn(b, nmax, dx, device=device)
 150              e = torch.randn(b, nmax, nmax, de, device=device)
 151              e = (e + e.transpose(1, 2)) / 2
 152              x = x * mask[..., None]
 153              e = e * pair_mask[..., None]
 154              def velocity(t, values):
 155                  x, e = values
 156                  pred = model(x, e * pair_mask[..., None], torch.ones(b, 1, device=device) * t, mask)
 157                  return ((pred.X.softmax(-1) - x) / (1 - t) * mask[..., None],
 158                          (pred.E.softmax(-1) - e) / (1 - t) * pair_mask[..., None])
 159              with torch.no_grad():
 160                  if method == "euler":
 161                      dt = endpoint / steps
 162                      for step in range(steps):
 163                          vx, ve = velocity(step * dt, (x, e))
 164                          x, e = x + dt * vx, e + dt * ve
 165                  else:
 166                      from torchdiffeq import odeint
 167                      xx, ee = odeint(velocity, (x, e), torch.tensor([0., endpoint], device=device), method=method,
 168                                      atol=float(sample.get("atol", 1e-5)), rtol=float(sample.get("rtol", 1e-5)))
```

## S3 — external_cli.py (epoch and generation routing)
SHA256: `95ecb4e4c06a9190d9705b05aacfbe7edcb3b895c784d65b13baf51c1682693b`

Lines 40–42:
```python
  40      parser.add_argument("--epochs", "--num-epochs", "--n-epochs", dest="epochs", type=positive)
  41      parser.add_argument("--batch-size", type=positive)
  42      parser.add_argument("--generation-batch-size", type=positive)
```

Lines 85–110:
```python
  85      if args.stage in ("train", "all"):
  86          training_options = dict(options)
  87          train = {}
  88          for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size), ("log_every", args.epoch_progress_interval)):
  89              if value is not None: train[key] = value
  90          if train: training_options["train"] = train
  91          if args.max_nodes is not None: training_options["max_nodes"] = args.max_nodes
  92          result = wrapper.train(TrainRequest(run=run,
  93                  dataset=DatasetReference(args.dataset, root=args.dataset_root, serialized_id=args.serialized_dataset or profile.serialized_id),
  94                  config_path=config, options=training_options, overwrite=args.overwrite))
  95          checkpoint = result.checkpoint_path
  96          summary["training_manifest"] = str(result.manifest_path)
  97      else:
  98          checkpoint = args.checkpoint or run.layout.checkpoints_dir / (model + ".pt")
  99          # Runtime/sample options may be overridden at generation, but never
 100          # silently replace model hyperparameters stored with the checkpoint.
 101          raw = yaml.safe_load(config.read_text()) or {}
 102          section = raw.get(model, raw.get("gsdm", raw) if model == "gdsm" else raw)
 103          from grapher.models.external_wrapper import merge
 104          generated = {k: v for k, v in section.items() if k in {"runtime", "sample", "generation_batch_size"}}
 105          options = merge(generated, options)
 106      summary["checkpoint"] = str(checkpoint)
 107      if args.stage in ("generate", "all"):
 108          result = wrapper.generate(GenerateRequest(run=run, checkpoint_path=checkpoint, num_graphs=args.num_samples,
 109                       generation_seed=args.generation_seed if args.generation_seed is not None else args.seed_id,
 110                       generation_id=args.generation_id, options=options, overwrite=args.overwrite))
```

## S4 — uploaded VFM_CatFlow / utils.py (model inputs)
SHA256: `1025a738b7c3aee3d94d393831ec58bcdcd4699021b5d3a2587448e876c49464`

Lines 61–84:
```python
  61  def to_dense(x, edge_index, edge_attr, batch, max_nodes):
  62      x = x.long()
  63      edge_index = edge_index.long()
  64      edge_attr = edge_attr.long()
  65  
  66      X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_nodes)
  67  
  68  
  69      # node_mask = node_mask.float()
  70      edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
  71      E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_nodes)
  72      # E = encode_no_edge(E)
  73  
  74      m = E.sum(dim=3) == 0
  75      ten = torch.zeros(E.shape[-1], device=E.device)
  76      ten[0] = 1
  77  
  78      E[m] = ten.long()
  79  
  80      diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
  81      E[diag] = 0
  82  
  83      return PlaceHolder(X=X, E=E, y=None), node_mask
  84  
```

Lines 306–346:
```python
 306  def get_GT_model(args, node_feats, edge_feats):
 307      from models.transformer import GraphTransformer
 308  
 309      if args.small_model == 1:
 310          hidden_dims = {'dx': 16, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 16, 'dim_ffE': 8, 'dim_ffy': 8}
 311          hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 16}
 312      elif args.task == 'abstract':
 313          hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
 314          hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
 315      else:
 316          # old
 317          # hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
 318          # hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
 319          hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
 320          hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
 321  
 322  
 323      model = GraphTransformer(
 324          input_dims={'X': node_feats, 'E': edge_feats, 'y': 1},
 325          # input_dims={'X': node_feats + 3, 'E': edge_feats, 'y': 1 + 6},
 326          hidden_dims=hidden_dims,
 327          hidden_mlp_dims=hidden_mlp_dims,
 328          output_dims={'X': node_feats, 'E': edge_feats, 'y': 1},
 329          # output_dims={'X': node_feats - 1, 'E': edge_feats, 'y': 1},
 330          n_layers=args.num_layers,
 331          act_fn_in=nn.ReLU(),
 332          act_fn_out=nn.ReLU(),
 333      )
 334  
 335      return model
 336  
 337  def project_simplex(n):
 338      fac_1 = torch.sqrt(torch.tensor([1 + 1/n])) * torch.ones(n, n)
 339      fac_2 = torch.pow(torch.tensor([n]), -(3/2)) * torch.ones(n, n)
 340      fac_3 = (torch.sqrt(torch.tensor([n + 1])) + 1) * torch.ones(n, n)
 341  
 342      verts = fac_1 * torch.eye(n) - fac_2 * fac_3
 343      extra_vert = torch.ones(1, n) * torch.pow(torch.tensor([n]), -(1/2))
 344      verts = torch.cat((verts, extra_vert), 0)
 345      return verts
 346  
```
