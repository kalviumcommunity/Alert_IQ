# Sprint 2 RAG Final Delivery

## Delivery status

This document is the reproducibility checklist for the final Sprint 2 RAG submission.

### System flow

```text
Upload -> Load/Clean -> Chunk -> Embed -> ChromaDB
                                         |
Question -> Retrieve -> Ground -> Generate -> Answer + Citation
                           |
                    Cache / Usage Logs
```

## Acceptance checklist

- [x] Backend runnable with `python -m src.api`
- [x] Streamlit UI runnable with `streamlit run ui/streamlit_stream.py`
- [x] Environment template committed as `.env.example`
- [x] `.env` and `.env.local` excluded by `.gitignore`
- [x] Document upload endpoint available at `POST /documents`
- [x] Question endpoint available at `POST /query`
- [x] Streaming endpoint available at `POST /query/stream`
- [x] Grounded answers include retrieved source metadata
- [x] Citation/source inspection is available in the UI
- [x] Caching, logging, and usage monitoring are included
- [x] Reproducible demo document committed at `data/samples/refund-policy.md`
- [x] End-to-end smoke script committed at `scripts/e2e_demo.py`
- [x] Automated tests can be run with `pytest -q`

## End-to-end demo

From the repository root, with the environment configured and backend running:

```bash
python scripts/e2e_demo.py
```

The script checks backend health, uploads the refund policy, asks the refund-window question, and prints the returned answer and source citation.

Expected source evidence includes `refund-policy.md` and the refund-policy chunk stating that refund requests must be submitted within 14 days.

## Reviewer runbook

1. Clone the repository.
2. Create `.env` from `.env.example`.
3. Configure the provider API key and model settings locally.
4. Install `requirements.txt` and `ui/requirements.txt`.
5. Start `python -m src.api`.
6. Run `python scripts/e2e_demo.py`.
7. Start the Streamlit UI with `streamlit run ui/streamlit_stream.py`.
8. Upload a document and ask a question in the UI.
9. Expand Sources and verify the citation text.
10. Run `pytest -q`.

## Secret handling

No real credentials belong in Git. `.env.example` contains placeholders and `.gitignore` excludes local environment files. Hosted deployments should inject credentials through platform environment variables or secret management.

## Final version marker

The PR for this learning unit is the **final-delivery candidate**. After the PR is reviewed and merged, create the requested final Git tag from the resulting `main` commit:

```bash
git tag sprint-2-rag-final
git push origin sprint-2-rag-final
```

This sequencing prevents the final tag from pointing to an unreviewed feature branch commit.
