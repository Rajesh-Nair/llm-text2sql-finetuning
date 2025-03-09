# TinyLlama for Text-to-SQL

## Problem Statement

I need a small generative model that can generate SQL code in response to user queries while avoiding any additional commentary. This will help reduce operational costs, increase throughput, and lower latency.

## Solution

### Part 1: Initial Experimentation (Refer to `Run_Tinyllama_Chat.ipynb`)

#### Step 1: Using an Off-the-Shelf Model

I started with the TinyLlama model. Below is an example of the initial request and response:

```
<|system|>
CREATE TABLE head(age INTEGER)</s>
<|user|>
How many heads of the departments are older than 56?</s>
<|assistant|>
I don't have access to the latest data or the current headcount of the departments...
```

The model did not return the expected SQL query, which is understandable given the lack of context.

#### Step 2: Prompt Engineering

I attempted prompt engineering by adding more details to the context:

```
<|system|>
You can only reply in SQL query language. Provide only SQL for the user's query given this context --> CREATE TABLE head(age INTEGER)</s>
<|user|>
How many heads of the departments are older than 56?</s>
<|assistant|>
SELECT COUNT(*) FROM head WHERE age > 56
```

The model generated the SQL query but included additional commentary, which I wanted to avoid.

#### Step 3: Further Refinement

Despite additional prompt engineering efforts, the model still produced unwanted explanations:

```
<|assistant|>
To calculate the number of heads of the departments older than 56, you can use the following SQL query:

SELECT COUNT(*) FROM departments WHERE age > 56;

In the above query, "departments" is the name of the table and "age" is the column name...
```

This led me to consider fine-tuning the model.

---

### Part 2: Fine-Tuning the Model

I decided to fine-tune TinyLlama for better SQL-specific responses. Below are the steps to replicate the fine-tuning process.

#### Setup Environment and Run Fine-Tuning Job on RunPod.io

```bash
#!/bin/bash
pip install -q accelerate transformers peft deepspeed bitsandbytes --no-build-isolation
pip install trl==0.9.6
pip install packaging ninja
MAX_JOBS=16 pip install flash-attn==2.6.0.post1 --no-build-isolation
git clone https://github.com/Rajesh-Nair/llm-text2sql-finetuning
cd llm-text2sql-finetuning
accelerate launch --config_file "ds_z3_qlora_config.yaml" train.py run_config.yaml | tee accelerate_output.log
```

#### Key Components of Fine-Tuning

1. **Dataset**: Utilized `b-mc2/sql-create-context` from Hugging Face for fine-tuning. High-quality data is essential for improving model performance.
2. **Accelerate**: Leveraged `accelerate` to enhance training speed and minimize boilerplate code.
3. **Distributed Training**:
   - Deployed across two GPUs on a single node via RunPod.io.
   - Hardware specifications: L4 GPU, PyTorch 2.1, Python 3.10, CUDA 11.8 (Ubuntu image).
4. **QLoRA**:
   - Applied QLoRA for memory-efficient fine-tuning.
   - Configured LoRA with 8-rank matrices for all linear layers.
5. **DeepSpeed Zero3**: Implemented for optimized sharding of optimizers, gradients, and parameters - along with flash attention.
6. **Mixed Precision**: Utilized to accelerate training and improve GPU efficiency.
7. **Batch Size & Gradient Accumulation**:
   - Set batch size per device to 4.
   - Applied gradient accumulation every 2 steps for optimal performance.
   - Increasing batch size beyond this sometimes led to GPU communication bottlenecks.
8. **Gradient Clipping**: Enabled to prevent unexpected exploding gradients.
9. **Training Duration & Cost**:
   - Each epoch took approximately 1 hour.
   - Training was force-stopped after 3 epochs due to negligible improvements in training loss.
   - Total fine-tuning cost on RunPod: under \$3.
10. **Training Logs**: Captured logs in `accelerate_outlog.log` for future analysis and reference.

#### Serving the Fine-Tuned Model

Refer to `Run_ft_Tinyllama_Chat.ipynb` for deploying the fine-tuned model.

Example Query and Response:

```
<|system|>
CREATE TABLE head(age INTEGER)</s>
<|user|>
How many heads of the departments are older than 56?</s>
<|assistant|>
SELECT COUNT(*) FROM head WHERE age > 56
```

The fine-tuned model now returns only the SQL query, as intended.

---

### Final Model & Deployment

After fine-tuning, I merged the trained adapter with the base model and uploaded it to Hugging Face: 🔗 [**TinyLlama-1.1B-Chat-Text2SQL-v1.0**](https://huggingface.co/mirajnair/TinyLlama-1.1B-Chat-Text2SQL-v1.0)

