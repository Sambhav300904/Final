# build_user_profile

Telemetry-only engineer profiles and provisional scoring against the R Systems framework (43 items).

**Sync first** (unchanged, separate entrypoint):

```powershell
cd ..\cursor_mysql_sync
python main.py --days 30
```

## Setup

```powershell
cd build_user_profile
pip install -r requirements.txt
```

Uses **`AI Adoption V1/.env`** at the repo root (same file as Cursor sync). Optional legacy `cursor_mysql_sync/.env` only fills missing keys.

```env
# See repo-root .env.example — multimodal profile (Qwen-VL, same token as evidence):
PROFILE_LLM_PROVIDER=multimodal
PROFILE_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
HF_TOKEN=<your_hf_token>

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSIONS=384
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=aiev-competencies-bge
TELEMETRY_MATCH_THRESHOLD=0.68
```

Set `PROFILE_LLM_PROVIDER=template` or pass `--no-llm` to skip the LLM.

## Files

| File | Role |
|------|------|
| `fetch_user_data.py` | MySQL → telemetry context JSON |
| `build_merged_context.py` | Cursor telemetry + evidence + Copilot MySQL → merged context |
| `fetch_copilot_data.py` | Copilot usage from `fact_copilot_*` tables |
| `create_profile.py` | Merged context → LLM/template profile |
| `embed_profile.py` | Profile narrative → vector |
| `competencies/ingest_competencies.py` | 43 items → Pinecone (run once) |
| `compare/compare_competencies.py` | Vector → scores + matched items |
| `competencies_data.py` | Framework checklist text |
| `pipeline_runner.py` | Same 4-step pipeline as Streamlit (subprocess) |
| `profile_store.py` | MySQL snapshots + roster queries |

Place evidence files under `../evidence_evaluator/evidences/<email>/` (PDF, DOCX, images, etc.). If the folder is missing or empty, the profile uses telemetry only.

Outputs go to `output/<email>/`.

## Run from repo root

```powershell
cd "C:\Users\Shubham.Gatkal\Desktop\AI Adoption V1"

python main.py ingest
python main.py all --user engineer@company.com --days 30
python batch_pipeline.py --days 7 --workers 4 --require-events
```

Batch reads `dim_cursor_team_members`, runs the same pipeline per user in parallel, writes `fact_user_proficiency_snapshot`.

## Run scripts directly

```powershell
cd build_user_profile
python fetch_user_data.py --user engineer@company.com --days 30
python create_profile.py --user engineer@company.com --days 30
python embed_profile.py --user engineer@company.com
python compare/compare_competencies.py --user engineer@company.com
python competencies/ingest_competencies.py
```

Scores are **provisional** (telemetry similarity only), not official framework checkbox credit.
