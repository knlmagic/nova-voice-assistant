"""
Configuration management for Nova Voice Assistant.

Loads settings from ~/.nova/config.toml with fallback to defaults.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "vad_threshold_seconds": 0.8,
        "enable_chimes": True,
        "enable_push_to_talk": True,
        "enable_vad": True
    },
    "stt": {
        "model_name": "small.en",
        "device": "auto",
        "language": "en"
    },
    "llm": {
        "base_url": "http://localhost:11434",
        "model": "mistral:7b-instruct-q4_K_M",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 150
    },
    "tts": {
        "device": "cpu",
        "voice_name": "default"
    },
    "memory": {
        "max_turns": 8,
        "db_path": "ai_memory.db"
    },
    "ui": {
        "verbose": False,
        "quiet": False,
        "profile": False
    }
}

@dataclass
class NovaConfig:
    """Main configuration class for Nova Assistant."""
    audio: Dict[str, Any]
    stt: Dict[str, Any] 
    llm: Dict[str, Any]
    tts: Dict[str, Any]
    memory: Dict[str, Any]
    ui: Dict[str, Any]
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'NovaConfig':
        """
        Load configuration from file with fallback to defaults.
        
        Args:
            config_path: Path to config file (defaults to ~/.nova/config.toml)
            
        Returns:
            NovaConfig instance with merged settings
        """
        # Determine config file path
        if config_path is None:
            config_path = Path.home() / ".nova" / "config.toml"
        else:
            config_path = Path(config_path)
        
        # Start with defaults
        config_data = DEFAULT_CONFIG.copy()
        
        # Try to load and merge user config
        if config_path.exists():
            try:
                import tomllib  # Python 3.11+
                with open(config_path, "rb") as f:
                    user_config = tomllib.load(f)
                
                # Deep merge user config over defaults
                config_data = _deep_merge(config_data, user_config)
                logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Config file not found at {config_path}, using defaults")
            
            # Create config directory and example file
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                _create_example_config(config_path)
            except Exception as e:
                logger.warning(f"Could not create example config: {e}")
        
        return cls(**config_data)
    
    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save config (defaults to ~/.nova/config.toml)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if config_path is None:
            config_path = Path.home() / ".nova" / "config.toml"
        else:
            config_path = Path(config_path)
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as TOML
            config_dict = asdict(self)
            toml_content = _dict_to_toml(config_dict)
            
            with open(config_path, "w") as f:
                f.write(toml_content)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def _create_example_config(config_path: Path) -> None:
    """Create an example configuration file."""
    toml_content = _dict_to_toml(DEFAULT_CONFIG)
    
    header = """# Nova Voice Assistant Configuration
# This file was auto-generated with default values.
# Customize settings below and restart Nova to apply changes.

"""
    
    with open(config_path, "w") as f:
        f.write(header + toml_content)
    
    logger.info(f"Created example config at {config_path}")

def _dict_to_toml(data: Dict[str, Any], indent: int = 0) -> str:
    """Convert dictionary to TOML format (simple implementation)."""
    lines = []
    indent_str = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}[{key}]")
            lines.append(_dict_to_toml(value, indent + 1))
        elif isinstance(value, str):
            lines.append(f'{indent_str}{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f'{indent_str}{key} = {str(value).lower()}')
        elif isinstance(value, (int, float)):
            lines.append(f'{indent_str}{key} = {value}')
        else:
            lines.append(f'{indent_str}{key} = "{str(value)}"')
    
    return "\n".join(lines)

# Convenience function
def load_config(config_path: Optional[str] = None) -> NovaConfig:
    """Load Nova configuration from file or defaults."""
    return NovaConfig.load(config_path)

if __name__ == "__main__":
    """Test configuration loading."""
    config = load_config()
    print("Loaded configuration:")
    print(f"Audio sample rate: {config.audio['sample_rate']}")
    print(f"LLM model: {config.llm['model']}")
    print(f"STT model: {config.stt['model_name']}") 