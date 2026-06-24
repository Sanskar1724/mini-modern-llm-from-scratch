# Mini Modern LLM - Technical Documentation

## Project Overview

Mini Modern LLM is a decoder-only Transformer language model implemented entirely from scratch using PyTorch. The goal of this project is to understand the core architectural components used in modern Large Language Models (LLMs) such as LLaMA, Gemma, Mistral, and Qwen.

Unlike traditional educational transformer implementations, this project incorporates several state-of-the-art improvements commonly found in modern production-grade language models.

The model is trained on the Tiny Shakespeare dataset using a next-character prediction objective.

---

# Objectives

The primary objectives of this project are:

* Understand transformer architecture from first principles.
* Implement modern LLM building blocks from scratch.
* Learn how attention mechanisms work internally.
* Explore positional encoding techniques.
* Understand normalization strategies used in modern models.
* Train a language model end-to-end.
* Generate Shakespeare-style text.
* Build a strong foundation for future work in LLM fine-tuning and deployment.

---

# Dataset

## Tiny Shakespeare Dataset

Source:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Statistics:

* Total Characters: ~1.1 Million
* Vocabulary Size: ~65 Unique Characters
* Task: Character-Level Language Modeling

Example:

ROMEO:
But, soft! what light through yonder window breaks?

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?

The model learns to predict the next character given a sequence of previous characters.

---

# Character Tokenization

The dataset is processed using character-level tokenization.

Example:

Input Text:

Hello

Encoded:

[20, 43, 50, 50, 53]

Each unique character is assigned an integer identifier.

Two mappings are created:

Character → Token ID

Token ID → Character

Functions:

encode(text)

decode(tokens)

---

# Data Pipeline

## Train Validation Split

The dataset is split into:

Training Data: 90%

Validation Data: 10%

Purpose:

Training Set:
Used for weight updates.

Validation Set:
Used for performance evaluation and overfitting detection.

---

## Batch Generation

Random batches are generated during training.

Input:

x

Target:

y

Target is simply the input shifted by one position.

Example:

Input:

HELLO

Target:

ELLO!

This teaches the model next-token prediction.

---

# Model Architecture

The model follows a Decoder-Only Transformer architecture.

Pipeline:

Input Tokens
→ Token Embedding
→ RoPE
→ Transformer Blocks
→ RMSNorm
→ LM Head
→ Vocabulary Logits

---

# RMSNorm

Root Mean Square Normalization

Formula:

RMS(x) = sqrt(mean(x²) + ε)

Output:

x / RMS(x)

Advantages:

* Simpler than LayerNorm
* No mean subtraction
* Faster computation
* Lower memory usage
* Used in LLaMA family models

Purpose:

Stabilizes activations during training.

---

# RoPE (Rotary Positional Embeddings)

Traditional transformers add positional embeddings.

RoPE instead rotates Query and Key vectors.

Benefits:

* Better long-context handling
* Captures relative positions naturally
* Improved extrapolation to unseen sequence lengths
* Used by LLaMA, Gemma, Qwen and Mistral

Core Idea:

Position information is encoded through rotation angles rather than additive vectors.

---

# Grouped Query Attention (GQA)

Standard Attention:

Q Heads = KV Heads

Grouped Query Attention:

Many Query Heads
Few Key/Value Heads

Example:

Q Heads = 8

KV Heads = 2

Each KV head is shared by multiple Query heads.

Benefits:

* Reduced memory usage
* Smaller KV cache
* Faster inference
* Maintains strong performance

Used in:

* LLaMA 2
* LLaMA 3
* Gemma
* Mistral

---

# SwiGLU Feed Forward Network

Traditional FFN:

Linear
→ GELU
→ Linear

Modern FFN:

Gate = SiLU(W_gate x)

Up = W_up x

Output:

Gate × Up

→ W_down

Advantages:

* Improved expressiveness
* Better gradient flow
* Higher model quality

Used in modern LLM architectures.

---

# Transformer Block

Each block consists of:

1. RMSNorm
2. Grouped Query Attention
3. Residual Connection
4. RMSNorm
5. SwiGLU
6. Residual Connection

Architecture:

Input
↓
RMSNorm
↓
GQA Attention
↓
Residual Add
↓
RMSNorm
↓
SwiGLU
↓
Residual Add

---

# Weight Tying

The token embedding matrix and output projection matrix share the same weights.

Benefits:

* Reduces parameter count
* Improves generalization
* Common practice in modern language models

---

# Training Configuration

Embedding Dimension: 256

Transformer Layers: 8

Query Heads: 8

KV Heads: 2

Feed Forward Dimension: 680

Context Length: 256

Batch Size: 32

Optimizer: AdamW

Learning Rate: 3e-4

Training Steps: 1000

---

# Loss Function

Cross Entropy Loss

Purpose:

Measures how well the predicted probability distribution matches the target token.

Lower loss indicates better next-token prediction performance.

---

# Training Workflow

1. Sample random batch
2. Compute forward pass
3. Calculate loss
4. Backpropagate gradients
5. Update parameters using AdamW
6. Evaluate periodically on validation data

---

# Text Generation

Generation is performed autoregressively.

Process:

1. Input prompt
2. Predict next character
3. Append prediction
4. Repeat

Sampling uses:

Temperature Scaling

Top-K Sampling

to improve generation diversity.

---

# Results

The model successfully learns:

* Character distributions
* English spelling patterns
* Shakespeare formatting
* Speaker names
* Punctuation structure

Example Output:

ROMEO:
O:
HA:
Tha owend tarer st betuertho...

Due to limited CPU training time, generated text remains partially nonsensical, but demonstrates successful learning of language structure.

---

# Visualizations

Included visualizations:

* RoPE Frequency Visualization
* Activation Function Comparison
* Training Loss Curves
* Weight Distribution
* Causal Attention Mask
* Model Architecture Diagram

---

# Key Learnings

This project provided hands-on experience with:

* PyTorch Fundamentals
* Transformer Internals
* Attention Mechanisms
* Positional Encoding
* Normalization Techniques
* Language Modeling
* Training Deep Neural Networks
* Modern LLM Design Principles

---

# Future Improvements

Potential future enhancements:

* Token-Level Language Modeling
* Flash Attention
* KV Caching
* LoRA Fine-Tuning
* Hugging Face Integration
* Streamlit Deployment
* Larger Datasets
* Mixed Precision Training
* Distributed Training

---

# Conclusion

This project demonstrates the implementation of a modern Transformer Language Model from scratch using PyTorch. By incorporating RMSNorm, RoPE, Grouped Query Attention, SwiGLU, and Weight Tying, the model closely mirrors architectural ideas used in contemporary Large Language Models.

The project serves as a strong educational foundation for understanding how modern LLMs are built, trained, and deployed.
