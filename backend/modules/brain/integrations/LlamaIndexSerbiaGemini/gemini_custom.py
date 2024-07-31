import typing
from typing import Any, Optional
from llama_index.llms.gemini import Gemini
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from google.generativeai.types import content_types
from llama_index.core.base.llms.types import (
    CompletionResponse
)

if typing.TYPE_CHECKING:
    import google.generativeai as genai

class GeminiCustom(Gemini):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        generation_config: Optional["genai.types.GenerationConfigDict"] = None,
        safety_settings: "genai.types.SafetySettingOptions" = None,
        callback_manager: Optional[CallbackManager] = None,
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
        model_name: Optional[str] = None,
        system_instructions: Optional[str] = None,
        **generate_kwargs: Any,
    ):
        """Creates a new GeminiCustom model interface with custom initialization."""
        
        # Call the parent class __init__ method
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            generation_config=generation_config,
            safety_settings=safety_settings,
            callback_manager=callback_manager,
            api_base=api_base,
            transport=transport,
            model_name=model_name,
            **generate_kwargs
        )
        
        # Custom initialization logic
        if system_instructions is None:
            self._model._system_instruction = None
        else:
            self._model._system_instruction = content_types.to_content(system_instructions)

    # You can also override other methods if needed
    # For example:
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        print("Custom complete method called")
        return super().complete(prompt, formatted=formatted, **kwargs)
