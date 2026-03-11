"""
Polish Social Economy LLM Benchmark Suite
==========================================
Evaluates LLM performance on 5 domain-specific tasks for the Polish social economy sector.

Tasks:
1. Grant Eligibility Assessment
2. E-Learning Engagement Strategy
3. Document Classification
4. CSR/ESG Report Generation
5. Text-to-SQL (Polish → PostgreSQL)

Hardware: NVIDIA RTX 4080 Laptop (12GB VRAM)
Target: NVIDIA DGX Spark (128GB) or RTX PRO 6000 (96GB)

Part of NVIDIA Academic Grant Program application.
"""

import json
import time
import urllib.request
import sys
from typing import List, Dict

OLLAMA_API = "http://localhost:11434/api"

BENCHMARK_SUITE = [
    {
        "id": "grant_eligibility_01",
        "category": "grant_eligibility",
        "prompt": "Organizacja pozarządowa z Krakowa (stowarzyszenie, KRS aktywny, 3 lata działalności, budżet 250 tys. PLN rocznie) chce złożyć wniosek o dotację na rozwój z programu FENG. Oceń kwalifikowalność i podaj wymagania.",
        "expected_keywords": ["KRS", "dotacja", "kwalifikowalność", "FENG", "wkład"],
        "description": "Assess NGO eligibility for EU funding programs"
    },
    {
        "id": "elearning_engage_01",
        "category": "elearning_engagement",
        "prompt": "Student ukończył 3 z 10 lekcji kursu online. Średni czas na lekcji: 4 minuty (norma: 12 min). Heartbeat aktywny tylko 40% czasu. Attention check: 1/3 poprawnych. Zaproponuj strategię zwiększenia zaangażowania.",
        "expected_keywords": ["zaangażowanie", "motywacja", "personalizacja", "quiz", "przypomnienie"],
        "description": "Generate personalized learning strategies from engagement metrics"
    },
    {
        "id": "doc_classify_01",
        "category": "document_classification",
        "prompt": "Sklasyfikuj dokument: 'Umowa nr 12/2026 o powierzenie grantu w ramach projektu \"Małopolska Przedsiębiorcza\" współfinansowanego ze środków EFS+. Kwota: 23 450,00 PLN.' Podaj typ dokumentu, program, kwotę.",
        "expected_keywords": ["umowa", "grant", "EFS", "23450", "Małopolska"],
        "description": "Extract structured information from Polish legal documents"
    },
    {
        "id": "csr_report_01",
        "category": "csr_report",
        "prompt": "Fundacja obsługuje 200+ organizacji, prowadzi kursy online (500 studentów), zarządza turnusami rehabilitacyjnymi i programami dotacyjnymi. Wygeneruj szkic raportu CSR/ESG za 2025 rok.",
        "expected_keywords": ["społeczny", "środowiskowy", "governance", "interesariusze", "wskaźniki"],
        "description": "Generate CSR/ESG report sections from operational data"
    },
    {
        "id": "text_to_sql_01",
        "category": "text_to_sql",
        "prompt": "Przekształć zapytanie na SQL (PostgreSQL): 'Pokaż mi wszystkie wnioski o dotację złożone w styczniu 2026, które mają status zaakceptowany i kwotę powyżej 10000 PLN, posortowane od najwyższej kwoty'",
        "expected_keywords": ["SELECT", "WHERE", "status", "ORDER BY", "DESC"],
        "description": "Convert Polish natural language to PostgreSQL queries"
    }
]


def query_model(model: str, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> Dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }).encode()

    start = time.time()
    req = urllib.request.Request(
        f"{OLLAMA_API}/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    elapsed = time.time() - start
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1)
    tok_per_sec = eval_count / (eval_duration / 1e9) if eval_duration else 0

    return {
        "response": data.get("response", ""),
        "time_sec": round(elapsed, 2),
        "tokens": eval_count,
        "tok_per_sec": round(tok_per_sec, 2),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
    }


def evaluate_response(response: str, expected_keywords: List[str]) -> Dict:
    hits = [kw for kw in expected_keywords if kw.lower() in response.lower()]
    misses = [kw for kw in expected_keywords if kw.lower() not in response.lower()]
    return {
        "accuracy": round(len(hits) / len(expected_keywords) * 100, 1),
        "hits": hits,
        "misses": misses
    }


def run_benchmark(models: List[str]) -> List[Dict]:
    all_results = []

    for model in models:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {model}")
        print(f"{'='*70}")

        for test in BENCHMARK_SUITE:
            print(f"\n  [{test['id']}] {test['description']}...")

            try:
                result = query_model(model, test["prompt"])
                eval_result = evaluate_response(result["response"], test["expected_keywords"])

                entry = {
                    "model": model,
                    "test_id": test["id"],
                    "category": test["category"],
                    "time_sec": result["time_sec"],
                    "tokens": result["tokens"],
                    "tok_per_sec": result["tok_per_sec"],
                    "keyword_accuracy": eval_result["accuracy"],
                    "keyword_hits": eval_result["hits"],
                    "keyword_misses": eval_result["misses"],
                    "response_length": len(result["response"]),
                    "response_preview": result["response"][:300]
                }
                all_results.append(entry)

                print(f"    Time: {result['time_sec']}s | {result['tok_per_sec']} tok/s | "
                      f"Accuracy: {eval_result['accuracy']}% | Tokens: {result['tokens']}")

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    "model": model,
                    "test_id": test["id"],
                    "error": str(e)
                })

    return all_results


def print_summary(results: List[Dict]):
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'Task':<25} {'Time(s)':<10} {'Tok/s':<10} {'Accuracy':<10}")
    print("-" * 75)

    for r in results:
        if "error" not in r:
            print(f"{r['model']:<20} {r['category']:<25} {r['time_sec']:<10} "
                  f"{r['tok_per_sec']:<10} {r['keyword_accuracy']:<10}")

    # Per-model averages
    models = set(r["model"] for r in results if "error" not in r)
    print(f"\n{'Model':<20} {'Avg Time':<12} {'Avg Tok/s':<12} {'Avg Accuracy':<12}")
    print("-" * 56)
    for model in sorted(models):
        model_results = [r for r in results if r.get("model") == model and "error" not in r]
        avg_time = sum(r["time_sec"] for r in model_results) / len(model_results)
        avg_toks = sum(r["tok_per_sec"] for r in model_results) / len(model_results)
        avg_acc = sum(r["keyword_accuracy"] for r in model_results) / len(model_results)
        print(f"{model:<20} {avg_time:<12.1f} {avg_toks:<12.1f} {avg_acc:<12.1f}")


if __name__ == "__main__":
    # Detect available models
    try:
        with urllib.request.urlopen(f"{OLLAMA_API}/tags") as resp:
            tags = json.loads(resp.read())
            available = [m["name"] for m in tags.get("models", [])]
    except:
        print("ERROR: Ollama not running")
        sys.exit(1)

    # Test priority: NVIDIA models first, then others
    test_models = []
    for m in available:
        if "nemotron" in m:
            test_models.insert(0, m)  # NVIDIA first
        elif "qwen3" in m:
            test_models.append(m)

    if not test_models:
        test_models = available[:3]

    print(f"Models to benchmark: {test_models}")
    results = run_benchmark(test_models)

    # Save
    with open("d:/tmp/grantos-ai-research/benchmarks/results/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_summary(results)
    print(f"\nResults saved to benchmarks/results/benchmark_results.json")
