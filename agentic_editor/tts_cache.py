from __future__ import annotations

import importlib
import typing

from agentic_editor.tss_cache import AbstractTtsCache

_DRIVER_FACTORIES: dict[str, str | type[AbstractTtsCache]] = {
    "poly": "agentic_editor.poly_tts_cache:PollyTtsCache",
    "polly": "agentic_editor.poly_tts_cache:PollyTtsCache",
    "qwen": "agentic_editor.qwen_tts_cache:QwenTtsCache",
}


def _get_config_value(
    config: dict[str, typing.Any], prefix: str, key: str
) -> object | None:
    normalized_prefix = prefix
    if normalized_prefix and not normalized_prefix.endswith("_"):
        normalized_prefix = f"{normalized_prefix}_"

    prefixed_key = f"{normalized_prefix}{key}" if normalized_prefix else key
    if prefixed_key in config:
        return config[prefixed_key]
    if key in config:
        return config[key]
    return None


def resolve_tts_cache_class(
    config: dict[str, typing.Any], prefix: str = ""
) -> type[AbstractTtsCache]:
    driver_name = _get_config_value(config, prefix, "speech_driver")
    if driver_name is None:
        driver_name = "poly"

    normalized_driver = str(driver_name).strip().lower()
    cache_class = _DRIVER_FACTORIES.get(normalized_driver)
    if cache_class is None:
        supported = ", ".join(sorted(_DRIVER_FACTORIES))
        raise ValueError(
            f"Unsupported speech_driver {driver_name!r}. Supported values: {supported}"
        )
    return _load_cache_class(cache_class)


def _load_cache_class(
    cache_class: str | type[AbstractTtsCache],
) -> type[AbstractTtsCache]:
    if not isinstance(cache_class, str):
        return cache_class

    module_name, _, class_name = cache_class.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Invalid TTS cache driver specifier: {cache_class!r}")

    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    if not isinstance(loaded_class, type):
        raise TypeError(f"TTS cache target {cache_class!r} is not a class")
    return typing.cast(type[AbstractTtsCache], loaded_class)


class TtsCache:
    """
    Stable cache entry point that delegates to a driver-specific implementation.
    """

    def __init__(
        self,
        config: dict[str, typing.Any],
        boto_session: object | None,
        prefix: str = "",
    ) -> None:
        cache_class = resolve_tts_cache_class(config, prefix=prefix)
        self._impl = cache_class(config, boto_session, prefix=prefix)

    def submit(
        self, text: str, *, priority: typing.Literal["background", "response"] = "background"
    ) -> None:
        self._impl.submit(text, priority=priority)

    def submit_ordered(
        self, texts: typing.Sequence[str], *, response_texts: typing.Sequence[str] = ()
    ) -> None:
        self._impl.submit_ordered(texts, response_texts=response_texts)

    def __getitem__(self, text: str):
        return self._impl[text]

    def __getattr__(self, name: str) -> object:
        return getattr(self._impl, name)


__all__ = ["TtsCache", "resolve_tts_cache_class"]
