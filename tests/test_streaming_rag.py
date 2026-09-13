import json

from src.streaming_rag import StreamingRAGService


class FakeChunk:
    id = "refund-policy-03"
    source_document = "refund-policy.md"
    chunk_index = 3
    score = 0.91
    text = "Refund requests must be submitted within 14 days."


class FakePipeline:
    dimension = 768
    store = object()
    client = object()


def decode(events):
    return [json.loads(event[6:]) for event in events if event.startswith("data: ")]


def test_sse_event_order_and_citation_content(monkeypatch):
    monkeypatch.setattr("src.streaming_rag.retrieve_stage", lambda **_: [FakeChunk()])
    monkeypatch.setattr("src.streaming_rag.assemble_stage", lambda chunks: chunks[0].text)
    monkeypatch.setattr("src.streaming_rag.generate_stage", lambda **_: "Grounded answer from the policy.")

    events = decode(list(StreamingRAGService(FakePipeline()).events("What is the refund policy?", delay_seconds=0)))

    assert events[0]["type"] == "citations"
    assert events[0]["sources"][0]["label"] == "[1]"
    assert events[0]["sources"][0]["document"] == "refund-policy.md"
    assert events[0]["sources"][0]["text"] == FakeChunk.text
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["type"] == "done"


def test_streaming_exception_emits_error(monkeypatch):
    def fail(**_):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr("src.streaming_rag.retrieve_stage", fail)
    events = decode(list(StreamingRAGService(FakePipeline()).events("hello", delay_seconds=0)))

    assert events[-1] == {
        "type": "error",
        "message": "The answer stopped streaming. Please retry.",
    }


def test_empty_question_emits_error():
    events = decode(list(StreamingRAGService(FakePipeline()).events("", delay_seconds=0)))
    assert events == [{"type": "error", "message": "Question cannot be empty."}]
