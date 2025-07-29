# Mr-Virgil
Code of [IROS 2025] "Mr. Virgil: Learning Multi-robot Visual-range Relative Localization"

![pipeline_page-0001](https://github.com/user-attachments/assets/b72f1d40-6597-4d17-9859-ae4986f0b351)

## Train & Fine-tuning & Evaluation & Inference
### Train
```python main.py --mode train```

### Fine-tuning
Freeze the match net parameters and continue training the pos net.

```python main.py --mode finetune```

### Evaluation
```python main.py --mode eval```

### Inference
You have to make sure:
- dataset have consecutive timestamps
- batch_size has been set to `1`

```python main.py --mode infer```

## Configs
Common parameters are listed in the `args.py`:
- `mode`: [train / finetune /eval /infer] 
- `epochs`: training epochs
- `batch_size`: batch size of the dataset
- `swarm_num`: robot numbers
- `max_cam_num`: max camera observations (should be larger than `swarm_num`)
- `lr`: learning rate
- `use_pgo`: if use differentiable pgo
- `shuffle`: if shuffle the dataset
- `model_file`: folder to load and save model checkpoints
- `train_dataset`: train dataset folder
- `val_dataset`: val dataset folder
- `embed_size`: dimension in the network
- `timestamp_thres`: allowed timestamp gap threshold between two frames when inference
