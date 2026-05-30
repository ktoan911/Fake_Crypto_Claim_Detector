---
title: Fake Crypto Claim Detector
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Fake Crypto Claim Detector

An AI system for detecting fake cryptocurrency and finance claims in Vietnamese, built on **Retrieval-Augmented Generation (RAG)**, a **LoRA-finetuned LLM**, and a **Confidence-Aware Fusion** classifier. The project ships with a FastAPI inference server, a Next.js web frontend, a browser extension, and a full data pipeline that crawls Vietnamese news sites and Facebook posts/groups.

## Highlights

- **Three-class verification** — every claim is classified as `A` (supported), `B` (refuted), or `C` (not enough evidence).
- **RAG over a Vietnamese knowledge base** — evidence is retrieved from an OpenSearch-backed news corpus using the `AITeamVN/Vietnamese_Embedding` retriever.
- **LoRA-finetuned LLM scorer** — a parameter-efficient adapter trained on Vietnamese fact-checking data produces the per-claim language signal.
- **Confidence-aware fusion** — a small fusion model combines retriever, LLM, and confidence features to make the final decision and to expose calibrated confidence scores.
- **Production-oriented API** — dynamic micro-batching, TTL cache, back-pressure on overload, and timeout-protected inference paths.
- **End-to-end stack** — crawlers, training scripts, REST API, web UI, and browser extension live in the same repo.

## Architecture

```
                ┌────────────────────┐
                │  Browser extension │
                │   / Next.js UI     │
                └─────────┬──────────┘
                          │  POST /verify
                          ▼
            ┌──────────────────────────────┐
            │   FastAPI (api_server.py)    │
            │  - TTL cache + back-pressure │
            │  - Dynamic micro-batching    │
            └─────────┬────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │      FusionClaimVerifier        │
        │                                 │
        │   ┌────────┐    ┌────────────┐  │
        │   │Retriever│──▶│ Evidence   │  │
        │   └────────┘    │ (top-k)    │  │
        │                 └─────┬──────┘  │
        │                       ▼         │
        │   ┌────────────────────────┐    │
        │   │ LoRA-finetuned LLM     │    │
        │   │   scorer (A/B/C)       │    │
        │   └─────────┬──────────────┘    │
        │             ▼                   │
        │   ┌────────────────────────┐    │
        │   │ Fusion classifier      │──▶ verdict + confidence
        │   └────────────────────────┘    │
        └─────────────────────────────────┘
                      ▲
                      │
        ┌─────────────────────────────────┐
        │  OpenSearch knowledge base      │
        │  (news_kb, stats)               │
        └─────────────────────────────────┘
                      ▲
                      │
        ┌─────────────────────────────────┐
        │  Crawlers: news + Facebook      │
        │  (src/data_process/crawlers)    │
        └─────────────────────────────────┘
```

## Project Structure

```
├── src/                       # Core library
│   ├── config.py              # Global config, label map, prompt templates
│   ├── utils.py               # Shared utilities
│   ├── llm_call.py            # LLM client wrapper
│   ├── llm_scorer.py          # LoRA-LLM inference / A-B-C scoring
│   ├── cluster_claims.py      # Claim clustering for stats / dedup
│   ├── models/
│   │   ├── fusion.py                # Fusion classifier definition
│   │   └── fusion_inference.py      # End-to-end FusionClaimVerifier
│   ├── training/
│   │   ├── lora_trainer.py          # PEFT/LoRA fine-tuning loop
│   │   └── fusion_trainer.py        # Fusion classifier training loop
│   ├── retrieval/
│   │   ├── embeddings.py            # Vietnamese embedding wrapper
│   │   └── retrieval.py             # KNN search over OpenSearch
│   ├── database/
│   │   └── opensearch.py            # OpenSearchKB client (news_kb, stats)
│   └── data_process/
│       ├── csv_loader.py            # Train/dev/test CSV utilities
│       └── crawlers/                # News crawler + Facebook scraper
│
├── scripts/                   # Entry-point scripts
│   ├── api_server.py                # FastAPI inference server
│   ├── cli.py                       # Crawl + verify CLI
│   ├── train_lora.py                # LoRA fine-tuning
│   ├── train_fusion.py              # Fusion model training
│   ├── train_retrieval.py           # Retrieval encoder training
│   ├── resume_train_fusion_from_hf.py
│   ├── upload_fusion_checkpoint_to_hf.py
│   ├── gendata.py                   # Synthetic data generation
│   ├── calculate_claims_stats.py    # Build dashboard stats
│   ├── mock_claims_data.py
│   ├── mock_stats_data.py
│   └── monitor_resources.py
│
├── extension/                 # Browser extension (manifest, content, background)
├── frontend/                  # Next.js web frontend
├── data/                      # Train / dev / test CSVs
├── docs/                      # Thesis report, training notes, pipeline diagrams
├── tests/                     # Unit & integration tests
├── Dockerfile
└── requirements.txt
```

## Setup

### Python backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Browser extension

Load `extension/` as an unpacked extension in Chrome / Edge (Developer Mode → "Load unpacked").

## Running the API

```bash
# Local dev
uvicorn scripts.api_server:app --host 0.0.0.0 --port 7860 --reload

# Or via Docker
docker build -t fake-crypto-claim-detector .
docker run -p 7860:7860 --env-file .env fake-crypto-claim-detector
```

### Endpoints

- `GET /health` — liveness probe; reports model load state, queue depth, and back-pressure limits.
- `POST /verify` — body `{"claim": "..."}`. Returns `verdict`, `status`, `evidence`, `source_links`, and `confidence`. Cached, batched, and timeout-protected.
- `GET /claims/stats?date=YYYY-MM-DD` — dashboard stats from the `stats` index. Falls back to the most recent record when `date` is omitted or missing.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FUSION_MODEL` | auto-resolved | Path or HF repo of the fusion checkpoint |
| `LLM_FINETUNE` | — | Path or HF repo of the LoRA-finetuned LLM |
| `RETRIEVER_MODEL` | `AITeamVN/Vietnamese_Embedding` | Sentence embedding model |
| `OPENSEARCH_INDEX_NAME` / `OP_KB_NAME` | `news_kb` | Evidence index |
| `OP_STATS_INDEX` | `stats` | Dashboard stats index |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `FUSION_LLM_EVIDENCE_TOP_K` | `3` | Evidence passages fed to the LLM |
| `MAX_PENDING_REQUESTS` | `64` | Back-pressure threshold (returns 503 over this) |
| `INFERENCE_TIMEOUT_S` | `120` | Per-request inference timeout |
| `BATCH_MAX_SIZE` | `8` | Max claims merged into a single batch |
| `BATCH_MAX_WAIT_MS` | `50` | Max micro-batching wait window |

## Training

```bash
# 1. Train the retrieval encoder
python scripts/train_retrieval.py

# 2. LoRA fine-tune the LLM scorer on the fact-checking dataset
python scripts/train_lora.py

# 3. Train the fusion classifier on top of frozen retriever + LLM features
python scripts/train_fusion.py

# Resume / sync fusion checkpoints from Hugging Face
python scripts/resume_train_fusion_from_hf.py
python scripts/upload_fusion_checkpoint_to_hf.py
```

Training data lives under `data/` and is loaded via `src/data_process/csv_loader.py`. Synthetic claims can be produced with `scripts/gendata.py`.

## Crawl + Verify CLI

```bash
# A .txt file with one Facebook page/group URL per line
python scripts/cli.py --url links.txt --count 20 --min_comments 5
```

The CLI scrapes posts from each Facebook page or group, runs them through `FusionClaimVerifier`, and writes results to `posts.json`.

## Stats pipeline

`scripts/calculate_claims_stats.py` aggregates verified claims into the `stats` OpenSearch index, which powers the dashboard exposed at `GET /claims/stats`. `scripts/mock_*` helpers generate placeholder data for local UI development.

## Tests

```bash
pytest tests/
```

## Documentation

- [docs/thesis_report.md](docs/thesis_report.md) — full technical report.
- [docs/crawl_pipeline.png](docs/crawl_pipeline.png) — data pipeline diagram.
- [docs/training_lora/](docs/training_lora/) and [docs/training_fusion/](docs/training_fusion/) — training notes and ablations.

## License

For research and educational use. See the thesis report for evaluation methodology and dataset details.
