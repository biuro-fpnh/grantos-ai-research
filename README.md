# GrantOS AI Research — Polish LLM for Social Economy

> Fine-tuning Large Language Models for Polish-language NGO Management and Adaptive E-Learning

[![NVIDIA NeMo](https://img.shields.io/badge/NVIDIA-NeMo-76B900?logo=nvidia)](https://developer.nvidia.com/nemo)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791?logo=postgresql)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This repository contains research code and benchmarks for the **NVIDIA Academic Grant Program** application. We investigate fine-tuning LLMs for two underserved domains:

1. **Adaptive E-Learning** — personalizing learning paths based on real-time engagement signals (heartbeat, attention checks, assignment quality)
2. **AI-Driven NGO Management** — automating grant eligibility assessment, document classification, and reporting for 200+ social economy organizations

## Platform: GrantOS

Our production platform serves 200+ organizations across Poland:
- 18+ product modules (CRM, LMS, RAG, chatbots, HR, grants)
- 50,000+ documents in pgvector knowledge base (768-dim embeddings)
- 12 AI chatbot deployments
- E-learning with engagement tracking (500+ students)
- 19+ Docker microservices in production

## NVIDIA Software Stack

| Tool | Usage |
|------|-------|
| **NVIDIA NeMo** | Fine-tuning pipeline (LoRA/QLoRA) |
| **NVIDIA NIM** | Containerized inference microservices |
| **NVIDIA TensorRT-LLM** | Optimized low-latency serving |
| **NVIDIA RAPIDS** | Data preprocessing & feature engineering |
| **Nemotron** | Base model for Polish-language fine-tuning |
| **CUDA 12.x + cuDNN** | GPU acceleration foundation |

## Preliminary Benchmarks

Tested on NVIDIA RTX 4080 Laptop (12GB VRAM):

| Model | Task | Tok/s | Keyword Accuracy |
|-------|------|-------|-----------------|
| Nemotron-Mini 4B | Grant Eligibility | ~30 | Baseline |
| Qwen3 14B | Grant Eligibility | ~15 | Good |
| Qwen3 32B | Grant Eligibility | ~2.5 | Best quality |

**Key finding**: 12GB VRAM insufficient for production deployment. Fine-tuning requires 48-128GB (DGX Spark or RTX PRO 6000).

## Repository Structure

```
├── benchmarks/           # Preliminary benchmark scripts
│   ├── polish_social_economy_bench.py
│   └── results/
├── data/                 # Sample data (anonymized)
│   └── sample_prompts.json
├── nemo/                 # NeMo fine-tuning configs
│   └── config.yaml
├── docs/                 # Technical reports
│   └── preliminary_report.md
└── README.md
```

## Research Team

- **PI**: Dr inż. Marta Woźniak-Zapór — Uniwersytet Ignatianum w Krakowie
- **Co-PI**: Dr Wojciech Huszlak — UKEN Kraków
- **Tech Lead**: Sebastian Adamczyk — Fundacja Promocji Nowej Huty
- **Industry Partner**: FPN + Instytut Nowych Technologii (INT)

## Publications (Planned)

1. "Fine-Tuning LLMs for Polish-Language Social Economy" — ACL/EMNLP
2. "Engagement-Aware Adaptive E-Learning with Local LLMs" — LAK/EDM
3. "RAG-Enhanced Grant Management for NGOs" — AAAI/IJCAI

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Acknowledgments

This research is supported by the **NVIDIA Academic Grant Program** (application pending).
Hardware: NVIDIA DGX Spark / RTX PRO 6000 (requested).
