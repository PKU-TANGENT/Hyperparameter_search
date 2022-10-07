# Huggingface Style Hyper-parameter search
Working example of Huggingface style hyper parameter search with ray-tune and `Trainer`.
The algorithm selected is `Population Based Training`.

Existing resources on integrating `hyper-parameter search` with `Huggingface` is limited and buggy. `PR`s are welcome. 
## Env
```bash
conda create -n hyper python=3.8.13 -y
conda activate hyper
pip install -r requirements.txt
```
## Issues 
1. Inadequate inspection of restart process. Undesired behavior might occur with `optimizer`, `scheduler` and `data_loader`.
2. `ray` does not keep the best performing checkpoint over the training course. This implies that only checkpoints that complete their whole designated training schedule would be available.