# Runtime Upload Policy

Uploaded Alert_IQ documents are indexed immediately after successful ingestion.
The document becomes available to the RAG query endpoint without restarting the API service.

For production uploads, large documents should be processed as background jobs and indexed in batches with retries.
