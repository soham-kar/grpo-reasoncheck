# LLM Fine-Tuning with GRPO (Group-Relative Policy Optimization)

This repository implements **Group-Relative Policy Optimization (GRPO)** to fine-tune a large language model (LLM) with parameter-efficient LoRA adapters. GRPO generates *multiple candidate outputs per prompt*, ranks them via a set of reward functions (format checks, numeric checks, and exact correctness), computes relative rewards, and updates the policy to prefer higher-ranked responses — removing the need for a separate value network.

## Table of Contents
1. [Project Overview](#1-project-overview)  
2. [What You’ll Build / Goals](#2-what-youll-build--goals)  
3. [Requirements & Dependencies](#3-requirements--dependencies)  
4. [Repository Structure](#4-repository-structure)  
5. [Data & Prompt Format](#5-data--prompt-format)  
6. [Reward Functions](#6-reward-functions)  
7. [Training Pipeline](#7-training-pipeline-from-the-notebook)  
8. [Quick Example Code Snippets](#8-quick-example-code-snippets)  
9. [Inference / Demo](#9-inference--demo)  
10. [Evaluation & Expected Behaviour](#10-evaluation--expected-behaviour)  
11. [Deployment](#11-deployment)  
12. [Tips & Troubleshooting](#12-tips--troubleshooting)  
13. [Further Work & Citations](#13-further-work--citations)  
14. [License](#14-license)

---

## 1. Project Overview
This project implements **Generative Reinforcement Policy Optimization (GRPO)** for fine-tuning Large Language Models (LLMs). Unlike traditional RLHF with PPO, GRPO removes the need for a separate value function by applying *group-relative ranking*. This simplifies training while retaining strong alignment performance.

The trained model learns to:  
- Generate multiple candidate responses.  
- Rank responses based on reward functions (e.g., correctness, format, style).  
- Update its policy to consistently produce higher-ranked answers.

Key points:
- Base model: `Qwen/Qwen2.5-3B-Instruct` (4-bit inference / LoRA).
- LoRA setup: rank `r=64`, `lora_alpha=64`, applied to projection modules.
- Dataset: `openai/gsm8k` (math reasoning).
- Run config: `num_generations=8`, `max_steps=2000` (~10 hours on L4 GPU).

---

## 2. What You’ll Build / Goals
- 🏗️ Train an LLM with **GRPO** from a HuggingFace base model.  
- 🎯 Define and use **custom reward functions**.  
- ⚡ Efficient training with **LoRA** + `vLLM`.  
- 🤖 Deploy and run inference demos interactively.  
- 📊 Evaluate and visualize training performance (loss, reward curves, etc.).  

---

## 3. Requirements & Dependencies

```bash
pip install unsloth vllm
pip install --upgrade pillow
pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
```

`requirements.txt` example:
```
torch>=2.0
unsloth>=0.x
vllm>=0.x
transformers>=4.x
datasets>=2.x
git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
wandb
accelerate
huggingface_hub
```

---

## 4. Repository Structure
```
llm-grpo/
├─ src/
│  ├─ train.py              # Training wrapper
│  ├─ infer.py              # Inference/demo script
│  └─ reward_funcs.py       # Reward functions
├─ notebooks/
│  └─ GRPO_Training.ipynb   # Original notebook
├─ configs/
│  └─ grpo.yaml             # Training config
├─ examples/
│  └─ sample_prompts.jsonl
├─ outputs/                 # Saved checkpoints
├─ requirements.txt
└─ README.md
```

---

## 5. Data & Prompt Format
**Dataset**: `openai/gsm8k`  

Each record maps to:
```json
{
  "prompt": [
    {"role": "system", "content": "Respond in the following format: <reasoning>...</reasoning><answer>...</answer>"},
    {"role": "user", "content": "<question text>"}
  ],
  "answer": "<reference answer>"
}
```

---

## 6. Reward Functions
Defined in `reward_funcs.py`:
- **xmlcount_reward_func**: Heuristic check for tags.
- **soft_format_reward_func**: Regex-based permissive structure check.
- **strict_format_reward_func**: Strict pattern match for reasoning + answer.
- **int_reward_func**: Rewards numeric `<answer>` outputs.
- **correctness_reward_func**: Exact-match correctness against reference answer.

---

## 7. Training Pipeline (from the notebook)
1. Load base model & tokenizer with `unsloth.FastLanguageModel`.
2. Apply LoRA with target projection modules.
3. Prepare GSM8K dataset with `<reasoning>/<answer>` enforced format.
4. Configure GRPO hyperparameters (`GRPOConfig`).
5. Train with `GRPOTrainer`.
6. Save LoRA adapters & merged model; optionally push to HuggingFace Hub.

---

## 8. Quick Example Code Snippets
```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Define training config
training_args = GRPOConfig(
    learning_rate=5e-6,
    num_generations=8,
    max_steps=2000,
    per_device_train_batch_size=1,
    output_dir="outputs",
    report_to="wandb"
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

---

## 9. Inference / Demo
```python
from unsloth import FastLanguageModel
from vllm import SamplingParams

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="YOUR_HF_MODEL",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

text = tokenizer.apply_chat_template([
    {"role": "system", "content": "Respond in <reasoning>/<answer> format."},
    {"role": "user", "content": "How many r's are in the word strawberry?"},
], tokenize=False, add_generation_prompt=True)

print(model.fast_generate(text, sampling_params=sampling_params)[0].outputs[0].text)
```

---

## 10. Evaluation & Expected Behaviour
- **Reward curves**: Log via wandb.
- **Exact-match correctness**: Higher after GRPO.
- **Qualitative**: Outputs adhere better to required format.

Expected: Fewer hallucinations, consistent `<reasoning>/<answer>` structure, improved integer-answer accuracy.

---

## 11. Deployment
- **Local**: run via CLI / notebook.  
- **API Server**: FastAPI/Flask.  
- **HuggingFace Spaces**: Gradio demo.  
- **Merged model push**:
```python
model.push_to_hub_merged("YOUR_HF_REPO", tokenizer, save_method="merged_16bit", token=HF_TOKEN)
```

---

## 12. Tips & Troubleshooting
- OOM → use `load_in_4bit=True`, reduce batch size.
- Diverging rewards → lower `learning_rate`, tune `max_grad_norm`.
- Slow candidate gen → reduce `num_generations` during debugging.
- Bad reward shaping → test functions individually.

---

## 13. Further Work & Citations
- Replace heuristics with learned reward models.
- Multi-objective GRPO with weighted signals.
- Evaluate on other datasets beyond GSM8K.

**References:**
- [GRPO Paper](https://arxiv.org/abs/2403.02993)
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)

---

## 14. License
MIT License.

