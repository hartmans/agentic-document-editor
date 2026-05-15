import base64

import numpy as np

from agentic_editor.gemma4 import Gemma4AsrReceiver
from langchain_core.messages import AIMessage


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list, dict]] = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if len(messages) == 2:
            return AIMessage(content="Hot words: Laura, Linda, balls, ball.")
        return AIMessage(
            content="Change Laura said drop the balls to Laura said drop the ball."
        )


def test_gemma4_uses_config_block_and_caches_document_analysis(monkeypatch):
    captured_init_kwargs = {}
    fake_model = FakeChatModel()

    def fake_init_chat_model(**kwargs):
        captured_init_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr("agentic_editor.gemma4.init_chat_model", fake_init_chat_model)

    receiver = Gemma4AsrReceiver(
        {
            "gemma4": {
                "model": "openai:custom-gemma",
                "base_url": "http://vllm.local/",
                "api_key": "not-secret",
                "temperature": 0.25,
                "extra_body": {"top_k": 20},
            }
        }
    )

    first = receiver.transcribe_audio(
        np.zeros(16, dtype=np.float32),
        document="Laura said drop the balls.",
    )
    second = receiver.transcribe_audio(
        np.ones(8, dtype=np.float32) * 0.1,
        document="Laura said drop the balls.",
    )

    assert first == "Change Laura said drop the balls to Laura said drop the ball."
    assert second == first
    assert captured_init_kwargs == {
        "model": "custom-gemma",
        "model_provider": "openai",
        "base_url": "http://vllm.local/",
        "api_key": "not-secret",
        "temperature": 0.25,
        "extra_body": {"top_k": 20},
    }

    assert len(fake_model.calls) == 3
    analysis_messages, analysis_kwargs = fake_model.calls[0]
    assert len(analysis_messages) == 2
    assert "Reference document:" in analysis_messages[1].content
    assert analysis_kwargs == {"max_tokens": 2048}

    final_messages, final_kwargs = fake_model.calls[1]
    assert len(final_messages) == 4
    assert final_messages[1].content == analysis_messages[1].content
    assert final_messages[2].content == "Hot words: Laura, Linda, balls, ball."
    audio_blocks = final_messages[3].content
    assert isinstance(audio_blocks, list)
    assert audio_blocks[0]["type"] == "text"
    assert "future editor model" in audio_blocks[0]["text"]
    assert audio_blocks[1]["type"] == "audio"
    assert audio_blocks[1]["mime_type"] == "audio/wav"
    assert isinstance(audio_blocks[1]["base64"], str)
    assert base64.b64decode(audio_blocks[1]["base64"])
    assert final_kwargs == {"max_tokens": 2048}


def test_gemma4_defaults_to_local_vllm_openai_adapter(monkeypatch):
    captured_init_kwargs = {}

    class MinimalChatModel:
        def invoke(self, messages, **kwargs):
            if len(messages) == 2:
                return AIMessage(content="Names: Gemma.")
            return AIMessage(content="Change Gemma to Gemma 4.")

    def fake_init_chat_model(**kwargs):
        captured_init_kwargs.update(kwargs)
        return MinimalChatModel()

    monkeypatch.setattr("agentic_editor.gemma4.init_chat_model", fake_init_chat_model)

    receiver = Gemma4AsrReceiver({})
    result = receiver.transcribe_audio(
        np.zeros(4, dtype=np.float32),
        document="Gemma",
    )

    assert result == "Change Gemma to Gemma 4."
    assert captured_init_kwargs["model"] == "google/gemma-4-E4B-it"
    assert captured_init_kwargs["model_provider"] == "openai"
    assert captured_init_kwargs["base_url"] == "http://127.0.0.1:8000/"
    assert captured_init_kwargs["api_key"] == "EMPTY"
    assert captured_init_kwargs["temperature"] == 0.0
