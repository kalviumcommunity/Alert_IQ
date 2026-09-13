from src.conversational_rag import ConversationalRAG


def test_history_tracks_user_and_assistant_messages():
    rag = ConversationalRAG("system", max_token_budget=200)
    rag.add_turn("What is replica lag?", "Check replication slots.")

    messages = rag.messages()
    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_follow_up_becomes_standalone_query():
    rag = ConversationalRAG("system")
    rag.add_turn(
        "PostgreSQL replica latency is 920ms. What should I check first?",
        "Check long-running transactions and replication slots.",
    )

    rewritten = rag.rewrite_follow_up("What's the next step?")

    assert "PostgreSQL replica latency" in rewritten
    assert "What's the next step?" in rewritten


def test_standalone_question_is_not_rewritten():
    rag = ConversationalRAG("system")
    rag.add_turn("What is replica lag?", "It is replication delay.")

    assert rag.rewrite_follow_up("How does PgBouncer work?") == "How does PgBouncer work?"


def test_retrieval_receives_rewritten_query():
    rag = ConversationalRAG("system")
    rag.add_turn("PostgreSQL replica latency is high.", "Check the connection pool.")
    seen = []

    def retrieve(query):
        seen.append(query)
        return ["PgBouncer pool utilization context"]

    result = rag.run(
        "What's the next step?",
        retrieve,
        lambda question, context: context[0],
    )

    assert seen == [result.rewritten_query]
    assert "PostgreSQL replica latency" in seen[0]
    assert result.answer == "PgBouncer pool utilization context"


def test_run_appends_follow_up_to_history():
    rag = ConversationalRAG("system")
    result = rag.run(
        "What should I check?",
        lambda query: ["verified context"],
        lambda question, context: "grounded answer",
    )

    assert result.history_messages[-2:] == [
        {"role": "user", "content": "What should I check?"},
        {"role": "assistant", "content": "grounded answer"},
    ]


def test_history_budget_trims_old_messages():
    rag = ConversationalRAG("system", max_token_budget=25)
    for i in range(8):
        rag.add_turn(f"Question {i} with extra detail", f"Answer {i} with extra detail")

    messages = rag.messages()
    assert messages[0]["role"] == "system"
    assert rag.history.count_total_tokens(messages) <= 25 or len(messages) <= 2
