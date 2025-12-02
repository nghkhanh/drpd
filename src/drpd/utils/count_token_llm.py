from typing import Optional
import tiktoken

class TokenCounter:
    """Handler token counting operations."""
    _encoders = {}

    def __init__(self):
        raise TypeError(" TokenCouter is a utility class and should not be instantiated")
    
    @classmethod
    def count_tokens(cls, text: str, model: Optional[str] = None) -> int:
        "Count token in text using OpenAI's tokenizer"
        if model not in cls._encoders:
            cls._encoders[model] = tiktoken.encoding_for_model(model)
        
        encoder = cls._encoders[model]

        try:
            tokens = encoder.encode(text)
            return len(tokens)
        except Exception as e:
            raise RuntimeError(f"Token counting failed: {e}") from e