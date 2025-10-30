# Vogent Turn Detection Demo

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Vogent Turn](https://img.shields.io/badge/vogent--turn-0.1.0-orange.svg)](https://github.com/vogent/vogent-turn)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive demonstration of **Vogent Turn Detection** - a state-of-the-art multimodal turn detection system for voice AI applications.

## Overview

This repository showcases the capabilities of Vogent Turn, a library that combines audio analysis and conversational context to accurately determine when a speaker has finished their turn in a conversation.

### Key Features

- **Multimodal Analysis**: Combines audio intonation (Whisper encoder) and text context (SmolLM)
- **High Accuracy**: State-of-the-art turn detection using prosodic, semantic, and contextual cues
- **Fast Inference**: Optimized with `torch.compile` for real-time applications
- **Production Ready**: Includes batch processing and comprehensive error handling
- **Easy Integration**: Simple Python API with minimal setup

## Architecture

```
Audio (16kHz) ──> Whisper Encoder ──> Audio Embeddings
                                            │
                                            ▼
Text Context ──> SmolLM Tokenizer ──> Text Embeddings
                                            │
                                            ▼
                                    Combined Processing
                                            │
                                            ▼
                                  Binary Classification
                                  (Complete / Incomplete)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using uv (faster):
```bash
uv pip install -r requirements.txt
```

## Quick Start

Run the main demonstration script:

```bash
python main.py
```

This will execute four comprehensive examples:
1. Simple turn detection
2. Batch processing
3. Real-world use cases
4. Technical advantages

## Usage Examples

### Basic Turn Detection

```python
from vogent_turn import TurnDetector
import soundfile as sf

# Initialize detector
detector = TurnDetector(compile_model=False)

# Load audio
audio, sr = sf.read("audio.wav")

# Detect turn endpoint
result = detector.predict(
    audio,
    prev_line="What is your phone number",
    curr_line="My number is 804",
    sample_rate=sr,
    return_probs=True
)

print(f"Turn complete: {result['is_endpoint']}")
print(f"Confidence: {result['prob_endpoint']:.1%}")
```

### Batch Processing

```python
# Process multiple audio files efficiently
results = detector.predict_batch(
    audio_batch,
    context_batch=context_batch,
    sample_rate=16000,
    return_probs=True
)
```

## Use Cases

### Voice Assistants
Accurately detect when users finish speaking to provide timely responses without interrupting.

**Example**: "Set alarm for..." [CONTINUE] → "...7 AM tomorrow" [COMPLETE]

### Call Center AI
Enable natural conversation flow by detecting turn boundaries automatically.

**Example**: "I need help with..." [CONTINUE] → "...my account" [COMPLETE]

### Meeting Transcription
Segment conversations accurately by identifying speaker turn boundaries.

**Example**: Speaker A finishes → [ENDPOINT] → Speaker B begins

### Voice Chat Bots
Respond with natural timing by understanding conversation flow.

**Example**: "My name is..." [CONTINUE] → "...John Smith" [COMPLETE] → Bot responds

### Language Learning
Evaluate pronunciation and sentence completion in educational applications.

**Example**: "The cat is..." [CONTINUE] → "...on the table" [COMPLETE]

## Audio Samples

This repository includes three audio samples for testing and demonstration:

### 1. Incomplete Number Sample
**File**: [`incomplete_number_sample.wav`](./incomplete_number_sample.wav) (Click to download)  
**Context**: "What is your phone number?" → "My number is 804"  
**Expected Result**: CONTINUE (speaker will continue)

### 2. Incomplete Response
**File**: [`incomplete.wav`](./incomplete.wav) (Click to download)  
**Context**: Partial response  
**Expected Result**: CONTINUE (speaker not finished)

### 3. Complete Number
**File**: [`complete.wav`](./complete.wav) (Click to download)  
**Context**: "What is your phone number?" → "My number is 8042221111"  
**Expected Result**: ENDPOINT (speaker finished)

---

These samples demonstrate the model's ability to distinguish between complete and incomplete utterances based on both audio cues (intonation, pauses) and conversational context.

### Testing with Audio Samples

```python
from vogent_turn import TurnDetector
import soundfile as sf

detector = TurnDetector(compile_model=False)

# Test with incomplete sample
audio, sr = sf.read("incomplete_number_sample.wav")
result = detector.predict(
    audio,
    prev_line="What is your phone number",
    curr_line="My number is 804",
    sample_rate=sr,
    return_probs=True
)
print(f"Incomplete: {result['is_endpoint']}")  # Expected: False

# Test with complete sample
audio, sr = sf.read("complete.wav")
result = detector.predict(
    audio,
    prev_line="What is your phone number",
    curr_line="My number is 8042221111",
    sample_rate=sr,
    return_probs=True
)
print(f"Complete: {result['is_endpoint']}")  # Expected: True
```

## Project Structure

```
.
├── main.py                      # Main demonstration script
├── tes.py                       # Additional test examples
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── incomplete_number_sample.wav # Audio sample: incomplete utterance
├── incomplete.wav               # Audio sample: partial response
├── complete.wav                 # Audio sample: complete utterance
├── vogent-turn/                 # Vogent Turn library source
│   ├── vogent_turn/            # Core library code
│   │   ├── __init__.py
│   │   ├── inference.py        # TurnDetector class
│   │   ├── predict.py          # CLI tool
│   │   ├── smollm_whisper.py   # Model architecture
│   │   └── whisper.py          # Whisper components
│   ├── examples/               # Usage examples
│   │   ├── basic_usage.py
│   │   ├── batch_processing.py
│   │   └── request_batcher.py
│   ├── pyproject.toml          # Package configuration
│   └── README.md               # Library documentation
└── venv/                        # Virtual environment (not in git)
```

## Technical Details

### Model Components

- **Audio Encoder**: Whisper-Tiny (processes up to 8 seconds of 16kHz audio)
- **Text Model**: SmolLM-135M (12 layers, approximately 80M parameters)
- **Classifier**: Binary classification head for turn detection

### Audio Requirements

- **Sample Rate**: 16kHz (automatically resampled if different)
- **Channels**: Mono
- **Format**: float32 numpy array
- **Range**: [-1.0, 1.0]
- **Duration**: Up to 8 seconds (longer audio is truncated)

### Performance Optimization

- Torch.compile support for faster inference
- Batch processing for efficient multi-sample analysis
- Model caching to reduce initialization overhead
- Automatic audio resampling when needed

## API Reference

### TurnDetector Class

```python
detector = TurnDetector(
    model_name="vogent/Vogent-Turn-80M",  # HuggingFace model ID
    revision="main",                       # Model revision
    device=None,                           # "cuda", "cpu", or None (auto)
    compile_model=True                     # Use torch.compile
)
```

### predict() Method

```python
result = detector.predict(
    audio,                    # np.ndarray: (n_samples,) mono float32
    prev_line="",             # str: Previous speaker's text
    curr_line="",             # str: Current speaker's text
    sample_rate=None,         # int: Sample rate in Hz
    return_probs=False        # bool: Return probabilities
)
```

**Returns**:
- If `return_probs=False`: `bool` (True = turn complete)
- If `return_probs=True`: `dict` with keys:
  - `is_endpoint`: bool
  - `prob_endpoint`: float (0-1)
  - `prob_continue`: float (0-1)

### predict_batch() Method

```python
results = detector.predict_batch(
    audio_batch,              # list[np.ndarray]: List of audio arrays
    context_batch=None,       # list[dict]: List of context dicts
    sample_rate=None,         # int: Sample rate in Hz
    return_probs=False        # bool: Return probabilities
)
```

## Development

### Running Tests

```bash
# Run main demonstration
python main.py

# Run additional tests
python tes.py
```

### Code Style

This project follows Python best practices:
- PEP 8 style guide
- Type hints where applicable
- Comprehensive docstrings
- Clear variable naming

## Resources

- [Vogent Turn GitHub](https://github.com/vogent/vogent-turn)
- [Technical Report](https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents)
- [Model Weights](https://huggingface.co/vogent/Vogent-Turn-80M)
- [HuggingFace Demo](https://huggingface.co/spaces/vogent/vogent-turn-demo)

## Citation

If you use Vogent Turn in your research or project, please cite:

```bibtex
@software{vogent_turn,
  title = {Vogent Turn: Multimodal Turn Detection for Conversational AI},
  author = {Vogent},
  year = {2024},
  url = {https://github.com/vogent/vogent-turn}
}
```

## License

This demonstration code is provided as-is for educational and research purposes.

Vogent Turn library:
- Inference code: Apache 2.0 License
- Model weights: Modified Apache 2.0 with attribution requirements

See the [vogent-turn LICENSE](vogent-turn/LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues related to:
- **This demo**: Open an issue in this repository
- **Vogent Turn library**: Visit [vogent-turn issues](https://github.com/vogent/vogent-turn/issues)

## Acknowledgments

- Vogent team for developing the turn detection library
- Whisper team at OpenAI for the audio encoder
- SmolLM team for the language model
- HuggingFace for model hosting and infrastructure

---

**Built with Vogent Turn** | **Powered by AI**
