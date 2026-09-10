from agentic_editor.graph import _build_llm


def test_build_llm_uses_default_llm_block(monkeypatch):
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("agentic_editor.graph.init_chat_model", fake_init_chat_model)

    _build_llm({})

    assert captured["model"] == "gemma4:31b"
    assert captured["model_provider"] == "ollama"
    assert len(captured["callbacks"]) == 1


def test_build_llm_uses_explicit_llm_block(monkeypatch):
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("agentic_editor.graph.init_chat_model", fake_init_chat_model)

    _build_llm(
        {
            "llm": {
                "model": "openai:gpt-4.1-mini",
                "base_url": "http://127.0.0.1:9000/v1",
                "api_key": "EMPTY",
                "temperature": 0.1,
            }
        }
    )

    assert captured["model"] == "gpt-4.1-mini"
    assert captured["model_provider"] == "openai"
    assert captured["base_url"] == "http://127.0.0.1:9000/v1"
    assert captured["api_key"] == "EMPTY"
    assert captured["temperature"] == 0.1
    assert len(captured["callbacks"]) == 1
