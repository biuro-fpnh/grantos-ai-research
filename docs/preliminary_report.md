# Preliminary Technical Report: LLM Feasibility for Polish Social Economy
## NVIDIA Academic Grant Program — Supporting Material

### Hardware
- NVIDIA RTX 4080 Laptop GPU (12GB VRAM, CUDA 12.x)
- Test environment: Windows 11, Ollama 0.17.6

### Models Tested
| Model | Parameters | VRAM Usage | Status |
|-------|-----------|------------|--------|
| Nemotron-Mini (NVIDIA) | 4.2B | ~3GB | Baseline — fastest |
| Qwen3 14B | 14B | ~9GB | Full GPU |
| Qwen3 32B | 32B | ~20GB | GPU+CPU offload |

### Benchmark: Polish Social Economy Tasks
5 task categories tested across all 3 models (15 total runs):

1. **Grant Eligibility Assessment** — classify NGO eligibility for EU programs
2. **E-Learning Engagement** — generate personalized learning strategies from engagement metrics
3. **Document Classification** — extract type, program, amount from legal documents
4. **CSR Report Generation** — generate ESG report sections from operational data
5. **Text-to-SQL** — convert Polish natural language to PostgreSQL queries

### Full Results: 3 Models x 5 Tasks

#### Nemotron-Mini (4.2B) — NVIDIA Baseline
| Task | tok/s | Accuracy | Notes |
|------|-------|----------|-------|
| Grant Eligibility | 10.0 | 20% | English-only response; recognized FENG only |
| E-Learning Engagement | 9.8 | 20% | Mixed language; hit 1/5 keywords |
| Document Classification | 10.73 | 40% | Concise; recognized umowa+EFS |
| CSR Report | 9.66 | 20% | English CSR template |
| Text-to-SQL | 11.33 | **100%** | Perfect SQL — best result overall |
| **Average** | **10.3** | **40%** | |

#### Qwen3 14B — Full GPU
| Task | tok/s | Accuracy | Notes |
|------|-------|----------|-------|
| Grant Eligibility | 5.25 | 0% | Thinking chain consumed all tokens |
| E-Learning Engagement | 4.96 | 0% | English thinking, Polish truncated |
| Document Classification | 4.52 | **80%** | Best classification result overall |
| CSR Report | 4.93 | 0% | Thinking chain consumed budget |
| Text-to-SQL | 4.99 | 0% | Polish table names instead of SQL |
| **Average** | **4.93** | **16%** | |

#### Qwen3 32B — GPU+CPU Offload
| Task | tok/s | Accuracy | Notes |
|------|-------|----------|-------|
| Grant Eligibility | 2.33 | 0% | 468s for 500 tokens |
| E-Learning Engagement | 2.37 | 20% | Partial GPU; 1/5 keywords |
| Document Classification | 2.54 | 0% | Thinking chain consumed output |
| CSR Report | 2.30 | 0% | Thinking chain consumed output |
| Text-to-SQL | 2.22 | 0% | Schema description instead of SQL |
| **Average** | **2.35** | **4%** | |

### Key Findings

#### 1. 12GB VRAM Is the Critical Bottleneck
- Nemotron-Mini (3GB) runs well but lacks Polish language capability
- Qwen3 14B (9GB) fits in VRAM but thinking chains waste the token budget
- Qwen3 32B (20GB) requires CPU offload — 2.3 tok/s is unusable for production
- No model >8B can be fine-tuned on 12GB (LoRA requires ~1.5x model size)

#### 2. Speed vs Quality Trade-off
- Nemotron-Mini: 10.3 tok/s average — fast but English-centric (needs Polish fine-tuning)
- Qwen3 14B: 4.93 tok/s — reasonable speed, best quality on structured tasks (80% doc classification)
- Qwen3 32B: 2.35 tok/s — theoretically best quality, but too slow to produce useful output

#### 3. Thinking Chain Problem (Qwen3)
- Both Qwen3 models generate internal "thinking" chains in English
- These thinking tokens consume the output budget before Polish content appears
- Result: 0% keyword accuracy on most tasks despite correct reasoning
- Solution requires either: `/no_think` mode, larger token budgets, or fine-tuning

#### 4. Task-Specific Insights
- **Text-to-SQL**: Language-agnostic task — Nemotron-Mini achieved 100% (SQL keywords are universal)
- **Document Classification**: Structured extraction favors larger models — Qwen3 14B achieved 80%
- **Open-ended tasks** (CSR, Grant, E-Learning): All models struggled with Polish keyword coverage

### GPU Bottleneck Analysis
Current RTX 4080 (12GB) limitations:
- Cannot fine-tune any model >8B parameters (LoRA requires ~1.5x model size)
- 32B model runs at 2.3 tok/s (CPU offload) — unsuitable for production
- Cannot serve inference + embeddings concurrently
- Single GPU cannot handle 12 concurrent chatbot sessions

**NVIDIA hardware (DGX Spark 128GB or RTX PRO 6000 96GB) would enable:**
- Fine-tuning 70B models with QLoRA (~48GB required)
- Production inference at 30+ tok/s for 32B model (full VRAM fit)
- Concurrent serving: inference + embeddings + 12 chatbots
- Training adaptive e-learning models on real engagement data
- Polish language fine-tuning of Nemotron family (best speed + custom Polish capability)

### Production Platform Statistics (GrantOS)
- 200+ organizations in CRM database
- 50,000+ documents in RAG knowledge base (pgvector 768-dim)
- 18+ product modules in production
- 12 AI chatbot deployments across landing pages
- 19+ Docker services running 24/7
- E-learning platform: heartbeat tracking, attention checks, creative assignments

### NVIDIA Software Stack (Current + Planned)
**Currently using:**
- CUDA 12.x (via Ollama GPU acceleration)
- cuDNN (implicit through PyTorch/Ollama)
- Nemotron-Mini as baseline inference model

**Planned integration (with grant hardware):**
- NVIDIA NeMo Framework — fine-tuning pipeline for Polish social economy tasks
- NVIDIA NIM — containerized inference microservices
- NVIDIA TensorRT-LLM — optimized serving (target: 30+ tok/s for 32B)
- NVIDIA RAPIDS — data preprocessing for 50k+ document corpus
- Pretrained models from ai.nvidia.com (Nemotron family — Polish fine-tuning)

### Conclusion
Full benchmark results (3 models x 5 tasks = 15 runs) confirm that 12GB VRAM is the critical bottleneck for production-grade Polish LLM deployment. The fastest model (Nemotron-Mini, 10.3 tok/s) lacks Polish capability; the most capable model (Qwen3 32B) is too slow at 2.3 tok/s due to CPU offload. NVIDIA Academic Grant hardware would directly resolve this by enabling:

1. **Polish fine-tuning** of Nemotron-Mini (combining speed + language capability)
2. **Full-VRAM inference** for 32B+ models (30+ tok/s vs current 2.3 tok/s)
3. **Concurrent production serving** for 200+ organizations and 12 chatbot instances

This research directly benefits the Polish social economy sector — 200+ NGOs, foundations, and social enterprises — while producing results publishable at NeurIPS, EMNLP, or ACL workshops on low-resource language NLP.
