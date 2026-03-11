# Preliminary Technical Report: LLM Feasibility for Polish Social Economy
## NVIDIA Academic Grant Program — Supporting Material

### Hardware
- NVIDIA RTX 4080 Laptop GPU (12GB VRAM, CUDA 12.x)
- Test environment: Windows 11, Ollama 0.17.6

### Models Tested
| Model | Parameters | VRAM Usage | Status |
|-------|-----------|------------|--------|
| Nemotron-Mini (NVIDIA) | 4.2B | ~3GB | Baseline |
| Qwen3 14B | 14B | ~9GB | Full GPU |
| Qwen3 32B | 32B | ~20GB | GPU+CPU offload |

### Benchmark: Polish Social Economy Tasks
5 task categories tested:
1. **Grant Eligibility Assessment** — classify NGO eligibility for EU programs
2. **E-Learning Engagement** — generate personalized learning strategies from engagement metrics
3. **Document Classification** — extract type, program, amount from legal documents
4. **CSR Report Generation** — generate ESG report sections from operational data
5. **Text-to-SQL** — convert Polish natural language to PostgreSQL queries

### Key Finding: GPU Bottleneck
Current RTX 4080 (12GB) limitations:
- Cannot fine-tune any model >8B parameters (LoRA requires ~1.5x model size)
- 32B model runs at 2.5 tok/s (CPU offload) — unsuitable for production
- Cannot serve inference + embeddings concurrently
- Single GPU cannot handle 12 concurrent chatbot sessions

**NVIDIA hardware (DGX Spark 128GB or RTX PRO 6000 96GB) would enable:**
- Fine-tuning 70B models with QLoRA (~48GB required)
- Production inference at 30+ tok/s for 32B model
- Concurrent serving: inference + embeddings + 12 chatbots
- Training adaptive e-learning models on real engagement data

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

**Planned integration (with grant hardware):**
- NVIDIA NeMo Framework — fine-tuning pipeline
- NVIDIA NIM — containerized inference microservices
- NVIDIA TensorRT-LLM — optimized serving
- NVIDIA RAPIDS — data preprocessing
- Pretrained models from ai.nvidia.com (Nemotron family)

### Conclusion
Preliminary results demonstrate the feasibility and urgent need for dedicated GPU resources.
Current hardware is insufficient for production-grade Polish LLM deployment in the social economy sector.
NVIDIA Academic Grant hardware would directly enable research publishable at top venues
while benefiting 200+ organizations in the Polish social economy.
