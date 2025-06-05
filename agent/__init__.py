"""
Agent package for voice assistant.

Provides LLM integration and conversation management.
"""

from .llm_client import (
    LLMConfig,
    LLMResponse,
    OllamaLLMClient,
    SystemPromptTemplate,
    ConversationContext,
    create_llm_client,
)

from .memory import (
    ConversationMessage,
    ConversationMemory,
    MemoryIntegratedLLMClient,
    create_memory_integrated_client,
)

__all__ = [
    "LLMConfig",
    "LLMResponse", 
    "OllamaLLMClient",
    "SystemPromptTemplate",
    "ConversationContext",
    "create_llm_client",
    "ConversationMessage",
    "ConversationMemory", 
    "MemoryIntegratedLLMClient",
    "create_memory_integrated_client",
] 