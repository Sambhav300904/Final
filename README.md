# AI Adoption V1

Entry points:

| Command | Purpose |
|---------|---------|
| `streamlit run streamlit_app.py` | **Web UI** — single user + **Batch (all users)** tab |
| `python cursor_mysql_sync/main.py --days 30` | Sync Cursor + Copilot → MySQL |
| `python main.py all --user you@rsystems.com --days 30` | One user: Cursor + evidence + Copilot → profile → score |
| `python batch_pipeline.py --days 7 --workers 4` | **All team members** (parallel) → MySQL snapshots |
| `python main.py batch --days 7 --workers 4 --require-events` | Same as `batch_pipeline.py` |

## Setup

```powershell
cd "C:\Users\Shubham.Gatkal\Desktop\AI Adoption V1"
copy .env.example .env
# Edit .env: Cursor, MySQL, Hugging Face (profile), Pinecone, HF_TOKEN (evidence)

pip install -r requirements.txt
pip install -r build_user_profile\requirements.txt
pip install -r evidence_evaluator\requirements.txt
pip install streamlit
```

**Config:** repo root `.env` — MySQL, `PROFILE_LLM_*` (Hugging Face router), `HF_TOKEN`, Pinecone.

## Web UI (Streamlit)

```powershell
streamlit run streamlit_app.py
```

- **Run pipeline** — one user (sidebar email)
- **Batch (all users)** — roster from `dim_cursor_team_members`, parallel workers, MySQL results
- For 800+ users, prefer CLI batch (see below)

## Batch pipeline (all users → MySQL)

**1. Create result tables (once):**

```powershell
python batch_pipeline.py --init-profile-schema --days 7
```

**2. Preview roster:**

```powershell
python batch_pipeline.py --days 7 --dry-run
python batch_pipeline.py --days 7 --dry-run --require-events
```

**3. Run (recommended: only members with Cursor events in period):**

```powershell
python batch_pipeline.py --days 7 --workers 4 --require-events
```

| Flag | Meaning |
|------|---------|
| `--workers 4` | Parallel users (4–6 typical with HF API; not 816 at once) |
| `--require-events` | Subset of roster with usage in the period |
| `--max-users 10` | Pilot run |
| `--skip-existing` | Skip users already stored for this period |
| `--no-evidence` | Telemetry-only (faster) |

**4. Query results:**

```sql
SELECT user_email, display_name, suggested_level_v1, telemetry_score,
       matched_count, profile_summary, pipeline_status
FROM fact_user_proficiency_snapshot
WHERE period_start = '2026-05-21' AND period_end = '2026-05-27'
ORDER BY telemetry_score DESC;
```

## Single-user pipeline (CLI)

```powershell
python main.py ingest
python main.py all --user shubham.gatkal@rsystems.com --days 30
```

See `build_user_profile/README.md` for details.

## Evidence uploads

1. `evidence_evaluator/evidences/<email>/` — PDF, DOCX, images, etc.
2. Included automatically in `main.py all` and batch pipeline (unless `--no-evidence`).

Requires `HF_TOKEN` (or `NOVITA_API_KEY`) in `.env`.
