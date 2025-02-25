
# Fine tunning
## Steps to run on runpod.io
```bash
#!/bin/bash
pip install -q accelerate transformers peft deepspeed bitsandbytes --no-build-isolation
pip install trl==0.9.6
pip install packaging ninja
MAX_JOBS=16 pip install flash-attn==2.6.0.post1 --no-build-isolation
git clone https://github.com/Rajesh-Nair/llm-text2sql-finetuning
cd llm-text2sql-finetuning
accelerate launch --config_file "ds_z3_qlora_config.yaml"  train.py run_config.yaml
```



# Plan

## Download and test a prompt (text to sql) with Tinyllama - 1 day

## Setup the data for Tinyllama text to sql fine tuning - 1 day

## Setup train.py and then accelerate with Deepspeed - 2 weeks

## Things to explore - 2 weeks
### Quanitization
### Lora
### DDP
### FDSP
## Distribute inference


# Note
## Build code and push different methods - under different repos