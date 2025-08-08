# grpo-reasoncheck
Training a reasoning-focused LLM using GRPO with final-answer verification to improve chain-of-thought accuracy.

# LLM Reasoning with GRPO

This repository reproduces the approach described in the transcript: using **Group Relative Policy Optimization (GRPO)** — a GRPO variant of PPO — to teach a language model to produce longer, more accurate chains of thought.

The core idea:

- For each prompt, **generate multiple candidate outputs** (`n` generations).
- Score each candidate using reward functions (e.g., correctness, format).
- Compute a **relative reward** (difference from mean) and rank candidates.
- Update the policy to prefer higher-ranked responses.

GRPO’s group-relative ranking removes the need for a large value network and simplifies training while encouraging reasoning behaviour.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [What You’ll Build / Goals](#what-youll-build--goals)  
3. [Requirements & Dependencies](#requirements--dependencies)  
4. [Repository Structure](#repository-structure)  
5. [Data & Prompt Format](#data--prompt-format)  
6. [Reward Functions](#reward-functions)  
7. [Training Pipeline](#training-pipeline)  
8. [Quick Example Code Snippets](#quick-example-code-snippets)  
9. [Inference / Demo](#inference--demo)  
10. [Evaluation & Expected Behaviour](#evaluation--expected-behaviour)  
11. [Deployment](#deployment)  
12. [Tips & Troubleshooting](#tips--troubleshooting)  
13. [Further Work & Citations](#further-work--citations)  
14. [License](#license)  

---

## Project Overview

This project demonstrates GRPO applied to a small open-source LLM (e.g., **Qwen-2.5-3B Instruct**) to improve verifiable reasoning such as GSM8K math problems.

Key implementation choices:

- **LoRA** parameter-efficient fine-tuning (rank `64` in example).
- Optional **4-bit loading** via `bitsandbytes`.
- `trl` for PPO/GRPO implementation.
- Reward function focused on **exact correctness**.
- Prompt template with `<think>` and `<answer>` tags.

---

## What You’ll Build / Goals

- A reproducible GRPO training recipe on verifiable problems.
- A LoRA-fine-tuned LLM that:
  - Generates reasoning inside `<think>` tags.
  - Outputs final answer inside `<answer>` tags.
  - Improves exact-answer accuracy vs. base model.
- A minimal demo notebook to run inference and compare before/after.

---

## Requirements & Dependencies

> Python 3.9+

Install dependencies:

```bash
pip install -U transformers datasets accelerate trl peft bitsandbytes evaluate huggingface_hub
# Optional
pip install wandb unsloth
