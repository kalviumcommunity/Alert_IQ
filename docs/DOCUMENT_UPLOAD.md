# Document Upload & Runtime Indexing

## Endpoint

`POST /documents` accepts a multipart/form-data upload using the `file` field.

Supported extensions are `.txt`, `.md`, and `.pdf` at the API boundary. The current text loader can directly parse the existing text/Markdown corpus formats; PDF uploads should be backed by a PDF text extractor when PDF ingestion is enabled in deployment.

## Flow

1. Validate the filename extension.
2. Read at most the configured upload limit plus one byte.
3. Reject empty or oversized uploads.
4. Strip path components and store the file under `UPLOAD_DIR`.
5. Load and normalize the document with `DocumentLoader`.
6. Clean the text with `TextCleaner`.
7. Token-chunk the cleaned document with the existing ingestion settings.
8. Generate deterministic embeddings through `CorpusIndexer.prepare_records`.
9. Upsert the records into the existing vector collection **without resetting it**.
10. Return the filename and indexing summary.

Because the endpoint performs an upsert into the same persistent vector collection used by `/query`, subsequent query requests can retrieve the newly indexed records without restarting the application.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `UPLOAD_DIR` | Upload storage directory | `uploads` |
| `UPLOAD_MAX_BYTES` | Maximum upload size | `5242880` |
| `EMBEDDING_DIMENSION` | Vector dimension | `768` |
| `INGESTION_CHUNK_SIZE` | Chunk size | `400` |
| `INGESTION_CHUNK_OVERLAP` | Chunk overlap | `60` |
| `VECTOR_DB_PERSIST_DIR` | Vector DB location | `data/vector_store` |
| `VECTOR_COLLECTION_NAME` | Vector collection | `alert_iq_knowledge_base` |

## Sample request

```bash
curl -X POST http://localhost:5000/documents \
  -F "file=@data/samples/runtime_policy.md"
```

Expected shape:

```json
{
  "status": "indexed",
  "filename": "runtime_policy.md",
  "summary": {
    "document": "uploads/runtime_policy.md",
    "chunks": 1,
    "indexed": 1,
    "collection": "alert_iq_knowledge_base"
  }
}
```

Then query the same API:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Do uploaded documents become searchable without restarting the API?"}'
```

See `logs/document_upload_demo.json` for the committed sample upload/index/follow-up evidence.

## Error behavior

- `400`: missing file or missing filename.
- `413`: upload exceeds `UPLOAD_MAX_BYTES`.
- `415`: unsupported extension or invalid/empty document.
- `500`: unexpected indexing failure.

## Large documents

For very large files, do not keep the complete ingestion operation inside the HTTP request. Persist the upload first, enqueue a background job, process the document in batches, retry failed embedding/indexing batches, and expose job status/progress to the client. This prevents request timeouts and keeps memory usage bounded.
