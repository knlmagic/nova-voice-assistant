"""
LLM Client for Mistral 7B integration via Ollama.

This module provides the core LLM interface for the voice assistant,
implementing the Nova persona and conversation management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ollama


logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model_name: str = "mistral:7b-instruct-q4_K_M"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 150  # Keep responses concise per PRD
    timeout: float = 30.0
    host: str = "http://localhost:11434"


@dataclass
class LLMResponse:
    """Response from LLM including metadata."""
    content: str
    latency_ms: float
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class SystemPromptTemplate:
    """Manages Nova system prompt template per PRD §4."""
    
    TEMPLATE = """You are **Nova**, a privacy-first personal voice assistant running entirely on my Mac.
– Keep responses under 30 words unless asked to elaborate.
– If unsure, ask clarifying questions.
– Never invent facts about personal data. If data not provided, say "I don't know yet".
– Use metric units unless user uses imperial first.
– Today's date is {current_date}."""

    @classmethod
    def format(cls, current_date: Optional[str] = None) -> str:
        """Format system prompt with current date."""
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        return cls.TEMPLATE.format(current_date=current_date)


class ConversationContext:
    """Manages conversation memory for LLM context injection."""
    
    def __init__(self, max_turns: int = 8):
        """Initialize with sliding window memory (last 8 turns per PRD)."""
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, content: str) -> None:
        """Add user message to conversation history."""
        self.messages.append({"role": "user", "content": content})
        self._maintain_window()
    
    def add_assistant_message(self, content: str) -> None:
        """Add assistant response to conversation history."""
        self.messages.append({"role": "assistant", "content": content})
        self._maintain_window()
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get conversation context for LLM prompt."""
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def _maintain_window(self) -> None:
        """Maintain sliding window of last N turns."""
        # Keep pairs of messages (user + assistant) up to max_turns
        if len(self.messages) > self.max_turns * 2:
            # Remove oldest pair
            self.messages = self.messages[2:]


class OllamaLLMClient:
    """
    LLM client using ollama-python for Mistral 7B integration.
    
    Provides async interface for generating responses with Nova persona.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize Ollama LLM client."""
        self.config = config or LLMConfig()
        self.client = ollama.Client(host=self.config.host)
        self.context = ConversationContext()
        self.prompt_template = SystemPromptTemplate()
        
        logger.info(f"Initialized OllamaLLMClient with model: {self.config.model_name}")
    
    async def generate_response(self, user_input: str) -> LLMResponse:
        """
        Generate response for user input with conversation context.
        
        Args:
            user_input: User's transcribed speech
            
        Returns:
            LLMResponse with content and metadata
        """
        start_time = time.time()
        
        try:
            # Add user message to context
            self.context.add_user_message(user_input)
            
            # Build messages with system prompt
            messages = [
                {"role": "system", "content": self.prompt_template.format()}
            ]
            messages.extend(self.context.get_context_messages())
            
            # Run Ollama chat in thread pool to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._ollama_chat_sync,
                messages
            )
            
            # Extract response content
            content = response.get("message", {}).get("content", "").strip()
            
            # Add assistant response to context
            self.context.add_assistant_message(content)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Build response object
            llm_response = LLMResponse(
                content=content,
                latency_ms=latency_ms,
                model=self.config.model_name,
                prompt_tokens=response.get("prompt_eval_count"),
                completion_tokens=response.get("eval_count")
            )
            
            logger.info(f"Generated response in {latency_ms:.1f}ms: {content[:50]}...")
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            
            # Return error response
            error_content = "I'm sorry, I'm having trouble processing that right now."
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=error_content,
                latency_ms=latency_ms,
                model=self.config.model_name
            )
    
    def _ollama_chat_sync(self, messages: List[Dict[str, str]]) -> Dict:
        """Synchronous Ollama chat call for thread executor."""
        return self.client.chat(
            model=self.config.model_name,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            }
        )
    
    async def health_check(self) -> Tuple[bool, str]:
        """
        Check if Ollama service and model are available.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            # Test connection and model availability
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.list()
            )
            
            # Check if our model is available
            model_names = [model.model for model in response.models]
            
            if self.config.model_name in model_names:
                return True, f"Model {self.config.model_name} available"
            else:
                return False, f"Model {self.config.model_name} not found. Available: {model_names}"
                
        except Exception as e:
            return False, f"Ollama service unavailable: {e}"
    
    def clear_conversation(self) -> None:
        """Clear conversation context."""
        self.context.clear()
        logger.info("Conversation context cleared")
    
    def get_stats(self) -> Dict:
        """Get current client statistics."""
        return {
            "model": self.config.model_name,
            "conversation_turns": len(self.context.messages) // 2,
            "max_turns": self.context.max_turns,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }


# Module-level convenience functions
async def create_llm_client(config: Optional[LLMConfig] = None) -> OllamaLLMClient:
    """Create and health-check LLM client."""
    client = OllamaLLMClient(config)
    
    is_healthy, status = await client.health_check()
    if not is_healthy:
        logger.warning(f"LLM client health check failed: {status}")
    else:
        logger.info(f"LLM client ready: {status}")
    
    return client


if __name__ == "__main__":
    """Test script for LLM client."""
    async def test_llm():
        # Create client
        client = await create_llm_client()
        
        # Test health check
        is_healthy, status = await client.health_check()
        print(f"Health check: {is_healthy} - {status}")
        
        if is_healthy:
            # Test conversation
            response = await client.generate_response("Hello, what's your name?")
            print(f"Response: {response.content}")
            print(f"Latency: {response.latency_ms:.1f}ms")
            
            # Test follow-up
            response2 = await client.generate_response("What can you help me with?")
            print(f"Follow-up: {response2.content}")
            print(f"Latency: {response2.latency_ms:.1f}ms")
            
            # Show stats
            print(f"Stats: {client.get_stats()}")
    
    # Run test
    import asyncio
    asyncio.run(test_llm()) 