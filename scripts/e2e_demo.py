"""Run the documented upload-to-answer smoke demo against a local API."""
import argparse
from pathlib import Path

import requests


DEFAULT_DOCUMENT = Path("data/samples/refund-policy.md")
DEFAULT_API_URL = "http://localhost:5000"
DEFAULT_QUESTION = "What is the refund window?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    document = args.document
    if not document.is_file():
        raise SystemExit(f"Demo document not found: {document}")

    health = requests.get(f"{api_url}/health", timeout=10)
    health.raise_for_status()
    print("[1/3] Backend healthy")

    with document.open("rb") as handle:
        upload = requests.post(
            f"{api_url}/documents",
            files={"file": (document.name, handle, "text/markdown")},
            timeout=120,
        )
    upload.raise_for_status()
    upload_data = upload.json()
    print(f"[2/3] Indexed: {upload_data['filename']}")

    query = requests.post(
        f"{api_url}/query",
        json={"question": args.question},
        timeout=120,
    )
    query.raise_for_status()
    result = query.json()

    print("[3/3] Grounded answer")
    print(result.get("answer", ""))
    print("\nSources:")
    for index, source in enumerate(result.get("sources", []), start=1):
        print(
            f"[{index}] {source.get('source_document', source.get('document', 'unknown'))} "
            f"· chunk {source.get('id', source.get('chunk_id', 'unknown'))} "
            f"· score {source.get('score', 0):.3f}"
        )


if __name__ == "__main__":
    main()
