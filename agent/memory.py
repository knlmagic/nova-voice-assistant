"""
Conversation memory system using SQLite for persistent storage.

Implements sliding window memory with last 8 turns per PRD specifications.
"""

import asyncio
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Individual conversation message with metadata."""
    id: Optional[int]
    role: str  # 'user' or 'assistant' 
    content: str
    timestamp: datetime
    
    @classmethod
    def from_row(cls, row: Tuple) -> 'ConversationMessage':
        """Create message from SQLite row."""
        return cls(
            id=row[0],
            role=row[1], 
            content=row[2],
            timestamp=datetime.fromisoformat(row[3])
        )


class ConversationMemory:
    """
    SQLite-based persistent conversation memory.
    
    Maintains sliding window of last 8 turns (16 messages) per PRD.
    Thread-safe for async usage.
    """
    
    def __init__(self, db_path: str = "ai_memory.db", max_turns: int = 8):
        """
        Initialize conversation memory.
        
        Args:
            db_path: Path to SQLite database file
            max_turns: Maximum conversation turns to maintain
        """
        self.db_path = Path(db_path)
        self.max_turns = max_turns
        self.max_messages = max_turns * 2  # user + assistant per turn
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        logger.info(f"Initialized ConversationMemory: {db_path}, max_turns={max_turns}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON conversations(timestamp DESC)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                yield conn
            finally:
                conn.close()
    
    async def add_message(self, role: str, content: str) -> int:
        """
        Add message to conversation memory.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            Message ID
        """
        def _add_sync() -> int:
            with self._get_connection() as conn:
                # Check for duplicate assistant messages
                if role == "assistant":
                    cursor = conn.execute("""
                        SELECT content FROM conversations 
                        WHERE role = 'assistant' 
                        ORDER BY timestamp DESC LIMIT 1
                    """)
                    row = cursor.fetchone()
                    if row and row[0].strip() == content.strip():
                        logger.debug(f"Skipping duplicate assistant message: {content[:30]}...")
                        return -1  # Indicate skipped
                
                cursor = conn.execute(
                    "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
                    (role, content, datetime.now().isoformat())
                )
                message_id = cursor.lastrowid
                conn.commit()
                
                # Maintain sliding window
                self._maintain_window(conn)
                
                return message_id
        
        # Run in thread pool to avoid blocking async event loop
        message_id = await asyncio.get_event_loop().run_in_executor(None, _add_sync)
        
        if message_id != -1:
            logger.debug(f"Added {role} message (ID={message_id}): {content[:50]}...")
        
        return message_id
    
    def _maintain_window(self, conn: sqlite3.Connection) -> None:
        """Maintain sliding window of recent messages."""
        # Count current messages
        cursor = conn.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        
        if count > self.max_messages:
            # Delete oldest messages beyond window
            messages_to_delete = count - self.max_messages
            conn.execute("""
                DELETE FROM conversations 
                WHERE id IN (
                    SELECT id FROM conversations 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            """, (messages_to_delete,))
            
            logger.debug(f"Deleted {messages_to_delete} old messages to maintain window")
    
    async def get_recent_messages(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """
        Get recent messages for conversation context.
        
        Args:
            limit: Maximum messages to return (defaults to sliding window)
            
        Returns:
            List of recent messages ordered by timestamp
        """
        if limit is None:
            limit = self.max_messages
        
        def _get_sync() -> List[ConversationMessage]:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, role, content, timestamp 
                    FROM conversations 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                """, (limit,))
                
                return [ConversationMessage.from_row(row) for row in cursor.fetchall()]
        
        messages = await asyncio.get_event_loop().run_in_executor(None, _get_sync)
        logger.debug(f"Retrieved {len(messages)} recent messages")
        return messages
    
    async def get_context_for_llm(self) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for LLM.
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = await self.get_recent_messages()
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    async def clear_conversation(self) -> int:
        """
        Clear all conversation history.
        
        Returns:
            Number of messages deleted
        """
        def _clear_sync() -> int:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                count = cursor.fetchone()[0]
                
                conn.execute("DELETE FROM conversations")
                conn.commit()
                
                return count
        
        deleted_count = await asyncio.get_event_loop().run_in_executor(None, _clear_sync)
        logger.info(f"Cleared conversation memory: {deleted_count} messages deleted")
        return deleted_count
    
    async def get_stats(self) -> Dict[str, any]:
        """Get memory statistics."""
        def _stats_sync() -> Dict[str, any]:
            with self._get_connection() as conn:
                # Message count
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                total_messages = cursor.fetchone()[0]
                
                # Turn count (pairs of user + assistant)
                cursor = conn.execute("SELECT COUNT(*) FROM conversations WHERE role = 'user'")
                user_messages = cursor.fetchone()[0]
                
                # Recent activity
                cursor = conn.execute("""
                    SELECT timestamp FROM conversations 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                last_activity = datetime.fromisoformat(row[0]) if row else None
                
                return {
                    "total_messages": total_messages,
                    "conversation_turns": user_messages,
                    "max_turns": self.max_turns,
                    "last_activity": last_activity,
                    "db_path": str(self.db_path)
                }
        
        return await asyncio.get_event_loop().run_in_executor(None, _stats_sync)
    
    async def vacuum_database(self) -> bool:
        """
        Vacuum the database to reclaim space and optimize performance.
        
        Returns:
            True if vacuum succeeded, False otherwise
        """
        def _vacuum_sync() -> bool:
            try:
                with self._get_connection() as conn:
                    conn.execute("VACUUM")
                    conn.commit()
                    logger.info("Database vacuum completed successfully")
                    return True
            except Exception as e:
                logger.error(f"Database vacuum failed: {e}")
                return False
        
        return await asyncio.get_event_loop().run_in_executor(None, _vacuum_sync)


class MemoryIntegratedLLMClient:
    """
    LLM client with integrated persistent memory.
    
    Combines SQLite memory with LLM conversation management.
    """
    
    def __init__(self, llm_client, memory_path: str = "ai_memory.db"):
        """
        Initialize memory-integrated LLM client.
        
        Args:
            llm_client: OllamaLLMClient instance
            memory_path: Path to SQLite memory database
        """
        self.llm_client = llm_client
        self.memory = ConversationMemory(memory_path)
        
        logger.info("Initialized MemoryIntegratedLLMClient")
    
    async def generate_response(self, user_input: str) -> 'LLMResponse':
        """
        Generate response with persistent memory context.
        
        Args:
            user_input: User's transcribed speech
            
        Returns:
            LLMResponse with content and metadata
        """
        # Store user message in persistent memory
        await self.memory.add_message("user", user_input)
        
        # Get conversation context from persistent memory
        context_messages = await self.memory.get_context_for_llm()
        
        # Override LLM client's in-memory context with persistent context
        self.llm_client.context.messages = context_messages
        
        # Generate response using LLM client
        response = await self.llm_client.generate_response(user_input)
        
        # Store assistant response in persistent memory
        await self.memory.add_message("assistant", response.content)
        
        return response
    
    async def clear_conversation(self) -> int:
        """Clear both persistent and in-memory conversation history."""
        deleted_count = await self.memory.clear_conversation()
        self.llm_client.clear_conversation()
        return deleted_count
    
    async def get_stats(self) -> Dict[str, any]:
        """Get combined memory and LLM statistics."""
        memory_stats = await self.memory.get_stats()
        llm_stats = self.llm_client.get_stats()
        
        return {
            **memory_stats,
            "llm_stats": llm_stats
        }


# Convenience functions
async def create_memory_integrated_client(llm_client, memory_path: str = "ai_memory.db"):
    """Create memory-integrated LLM client."""
    return MemoryIntegratedLLMClient(llm_client, memory_path)


if __name__ == "__main__":
    """Test script for conversation memory."""
    async def test_memory():
        # Test basic memory operations
        memory = ConversationMemory("test_memory.db")
        
        # Add test messages
        await memory.add_message("user", "Hello, what's your name?")
        await memory.add_message("assistant", "I'm Nova, your assistant.")
        await memory.add_message("user", "What can you help me with?")
        await memory.add_message("assistant", "I can help with various tasks.")
        
        # Get recent messages
        messages = await memory.get_recent_messages()
        print(f"Recent messages: {len(messages)}")
        for msg in messages:
            print(f"  {msg.role}: {msg.content}")
        
        # Get LLM context
        context = await memory.get_context_for_llm()
        print(f"LLM context: {context}")
        
        # Get stats
        stats = await memory.get_stats()
        print(f"Stats: {stats}")
        
        # Clean up test database
        import os
        await memory.clear_conversation()
        if os.path.exists("test_memory.db"):
            os.remove("test_memory.db")
    
    # Run test
    asyncio.run(test_memory()) 