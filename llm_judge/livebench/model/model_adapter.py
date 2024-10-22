import os
from functools import lru_cache
from typing import List

from fastchat.conversation import Conversation, get_conv_template


class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):  # pylint: disable=unused-argument
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        raise NotImplementedError

    def get_default_conv_template(self, model_path: str) -> Conversation:  # pylint: disable=unused-argument
        return get_conv_template("chatgpt")


model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@lru_cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)


register_model_adapter(BaseModelAdapter)
