# Nova Voice Assistant - Talk-Back MVP

A privacy-first voice assistant running entirely on your Mac with push-to-talk functionality.

## Features

- 🎤 **Push-to-Talk**: Press ⌃⌥␣ (Control-Option-Space) to talk
- 🗣️ **Speech-to-Text**: Powered by faster-whisper (small.en model)
- 🧠 **AI Responses**: Mistral 7B-Instruct via Ollama
- 🎵 **Text-to-Speech**: High-quality synthesis via Chatterbox TTS
- 💾 **Memory**: Persistent conversation history with SQLite
- 📊 **Performance Monitoring**: Real-time latency tracking
- 🔒 **Privacy-First**: Everything runs locally, no cloud dependencies

## Architecture

```
Push-to-Talk (⌃⌥␣) → Audio Capture → STT → LLM → TTS → Audio Output
                         ↑                              ↓
                   SQLite Memory ← ← ← ← ← ← ← ← ← ← Audio Chimes
```

## Prerequisites

- macOS (tested on macOS 14.5+)
- Python 3.11 (recommended via pyenv)
- Ollama with Mistral 7B model
- Homebrew (for system dependencies)

## Quick Setup

### 1. Install System Dependencies

```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Pull Mistral 7B model
ollama pull mistral:7b-instruct-q4_K_M

# Install audio dependencies
brew install portaudio ffmpeg
```

### 2. Setup Python Environment

```bash
# Clone the repository
cd voice-assistant-mvp

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Run the Assistant

```bash
# Activate environment
source venv/bin/activate

# Start Nova Assistant
python assistant.py

# Or use the launcher
python main.py
```

## Usage

### Basic Operation

1. **Start the assistant**: `python assistant.py`
2. **Wait for ready message**: "🎧 NOVA VOICE ASSISTANT READY"
3. **Press and hold ⌃⌥␣**: Control-Option-Space to start recording
4. **Speak your message**: While holding the hotkey
5. **Release the hotkey**: To stop recording and process
6. **Listen to response**: Nova will speak back to you

### Command Line Options

```bash
# Verbose mode (detailed logging)
python assistant.py --verbose

# Quiet mode (minimal output)
python assistant.py --quiet

# Show performance profile
python assistant.py --profile
```

### Example Conversation

```
🎤 Listening...
🗣️  You said: "Hello, how are you?"
🤖 Nova: "I'm just a computer program, but I'm glad to be here to help you!"
🔊 Speaking response...
```

## Testing

### Integration Tests

```bash
# Run comprehensive integration tests
python test_integration_full.py
```

### Component Tests

```bash
# Test individual components
python test_stt_basic.py
python test_audio_pipeline.py
python test_stt_llm_integration.py
```

## Performance

### Typical Latencies (M4 Pro Mac)

- **STT**: ~300-500ms
- **LLM**: ~600-2000ms (after warm-up)
- **TTS**: ~6-7s (CPU synthesis)
- **End-to-End**: ~7-10s total

### Memory Usage

- **Startup**: ~1.5GB
- **Runtime**: ~2GB sustained
- **Models**: ~6GB disk space (Mistral + Whisper + Chatterbox)

## Configuration

### Audio Settings

Edit `core/audio/pipeline.py`:
- Sample rate: 16kHz (default)
- VAD threshold: 0.8s silence
- Chunk size: 128ms

### LLM Settings

Edit `agent/llm_client.py`:
- Temperature: 0.7
- Top-p: 0.95
- Max tokens: 150
- Memory: 8 turns

### TTS Settings

Edit `core/tts/chatterbox_client.py`:
- Voice: resemble-ai/chatterbox-t3-en
- Sample rate: 24kHz
- Device: CPU (default)

## Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**"Ollama connection failed"**
```bash
# Check Ollama service
brew services restart ollama
ollama list  # Verify mistral model is available
```

**"Audio device not found"**
```bash
# Check audio permissions in System Preferences
# Restart the assistant
```

**"Global hotkey not working"**
```bash
# Grant accessibility permissions to Terminal
# System Preferences → Security & Privacy → Accessibility
```

### Performance Issues

**Slow TTS synthesis**
- TTS runs on CPU by default (~6-7s)
- This is expected for the MVP version
- Future versions will optimize for GPU acceleration

**High memory usage**
- Models are loaded in memory for performance
- Use `--quiet` mode to reduce logging overhead
- Restart assistant periodically for long sessions

## Development

### Project Structure

```
voice-assistant-mvp/
├── assistant.py              # Main orchestrator
├── main.py                   # Simple launcher
├── core/
│   ├── audio/               # Audio capture/playback
│   ├── stt/                 # Speech-to-text
│   └── tts/                 # Text-to-speech
├── agent/
│   ├── llm_client.py        # LLM integration
│   └── memory.py            # Conversation memory
├── utils/
│   ├── events.py            # Event system
│   ├── metrics.py           # Performance monitoring
│   └── device.py            # Device utilities
├── tests/                   # Test suite
└── sounds/                  # Audio chimes
```

### Adding Features

1. **New LLM Models**: Edit `agent/llm_client.py`
2. **Audio Processing**: Modify `core/audio/`
3. **Performance Metrics**: Extend `utils/metrics.py`
4. **Error Handling**: Update `utils/events.py`

## License

This project is part of the HomeAss voice assistant implementation.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the integration test output
3. Enable verbose mode for detailed logging
4. Check the `nova_assistant.log` file 