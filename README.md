# Video Content Safety Analyzer

A comprehensive video content analysis tool that combines visual text detection and audio transcription to identify potentially harmful or toxic content in videos. The system uses OCR (Optical Character Recognition) and Speech-to-Text technologies with multi-language toxicity detection.

## Features

- **Dual Analysis Pipeline**: Simultaneous visual text extraction and audio transcription
- **Multi-Language Toxicity Detection**: Supports English, Hindi, and Telugu
- **Smart False Positive Filtering**: Reduces incorrectly flagged content
- **Sentiment Analysis**: Evaluates emotional tone of detected content
- **Dynamic Frame Extraction**: Optimizes processing based on video duration
- **Comprehensive Reporting**: Detailed JSON output with confidence scores
- **Enterprise-Ready**: Configurable parameters and robust error handling

## Quick Start

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install tesseract-ocr ffmpeg

# For other systems, ensure you have:
# - Tesseract OCR
# - FFmpeg
# - Python 3.8+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-company/video-content-analyzer.git
cd video-content-analyzer
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download Whisper model (optional, will auto-download on first run):
```bash
python -c "import whisper; whisper.load_model('base')"
```

### Basic Usage

```bash
# Analyze a video file
python complete_analyzer_refactored.py video.mp4

# Use a specific Whisper model
python complete_analyzer_refactored.py video.mp4 large

# Analyze first video found in current directory
python complete_analyzer_refactored.py
```

## Architecture

### Core Components

- **`complete_analyzer_refactored.py`**: Main analysis engine
- **`config.py`**: Centralized configuration management
- **`CompleteVideoContentAnalyzer`**: Primary analysis class

### Analysis Pipeline

1. **Video Frame Extraction**: Dynamic FPS extraction based on duration
2. **Visual Text Detection**: EasyOCR + Tesseract fallback
3. **Audio Processing**: MoviePy + FFmpeg extraction
4. **Speech Transcription**: OpenAI Whisper models
5. **Content Analysis**: Toxicity detection + sentiment analysis
6. **Result Aggregation**: Combined safety assessment

## Configuration

All settings are managed through `config.py`:

### Key Configuration Areas

- **OCR Settings**: Confidence thresholds, languages
- **Video Processing**: Frame extraction rates, resolution limits
- **Audio Processing**: Sample rates, codec settings
- **Toxicity Detection**: Word lists, severity levels
- **Model Configuration**: Whisper model sizes, sentiment models

### Example Customization

```python
# config.py
OCR_CONFIDENCE_THRESHOLD = 0.5  # Increase for stricter text detection
MAX_VIDEO_WIDTH = 720           # Reduce for faster processing
DEFAULT_WHISPER_MODEL = "large" # Better accuracy, slower processing
```

## Output Format

The analyzer generates comprehensive JSON reports:

```json
{
  "content_safety": "safe|review_needed|unsafe",
  "toxicity_assessment": {
    "level": "safe|review_needed|unsafe",
    "confidence": "high|medium|low",
    "visual_score": 0.0,
    "audio_score": 0.0
  },
  "visual_analysis": {
    "detected_texts": ["text1", "text2"],
    "sentiment_analysis": {...},
    "toxicity_flags": {...}
  },
  "audio_analysis": {
    "transcript": "...",
    "transcription_confidence": 0.95,
    "sentiment_analysis": {...},
    "toxicity_flags": {...}
  }
}
```

## API Integration

### Basic Integration

```python
from complete_analyzer_refactored import CompleteVideoContentAnalyzer

# Initialize analyzer
analyzer = CompleteVideoContentAnalyzer(whisper_model_size="base")

# Analyze video
results = analyzer.process_video_complete("path/to/video.mp4")

# Check safety level
safety_level = results["content_safety"]
if safety_level == "unsafe":
    # Handle unsafe content
    pass
```

### Batch Processing

```python
import os
from pathlib import Path

analyzer = CompleteVideoContentAnalyzer()
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

for video_file in Path("videos/").glob("*"):
    if video_file.suffix.lower() in video_extensions:
        results = analyzer.process_video_complete(str(video_file))
        print(f"{video_file.name}: {results['content_safety']}")
```

## Performance Considerations

### Processing Speed

- **Short videos (<5s)**: ~10-30 seconds processing time
- **Medium videos (5-15s)**: ~30-90 seconds processing time
- **Long videos (>15s)**: ~1-5 minutes processing time

### Resource Usage

- **RAM**: 2-8GB depending on Whisper model size
- **GPU**: Optional, significantly improves performance
- **Storage**: Temporary audio files during processing

### Optimization Tips

1. **Use smaller Whisper models** for faster processing
2. **Reduce video resolution** in config for speed
3. **Enable GPU acceleration** for better performance
4. **Process videos in batches** for efficiency

## Security Considerations

### Data Privacy

- **No external API calls** - all processing is local
- **Temporary files are cleaned up** after processing
- **No data persistence** unless explicitly saved
- **Configurable logging levels** for audit trails

### Content Handling

- **Multi-language profanity filtering** with context awareness
- **False positive reduction** through contextual analysis
- **Confidence scoring** for manual review workflows
- **Detailed audit trails** in output reports

## Development

### Project Structure

```
video-content-analyzer/
├── complete_analyzer_refactored.py  # Main analysis engine
├── config.py                        # Configuration management
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── LICENSE                          # License information
├── tests/                           # Unit tests
├── examples/                        # Usage examples
└── docs/                           # Additional documentation
```

### Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Generate coverage report
python -m pytest --cov=complete_analyzer_refactored tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Dependencies

### Core Dependencies

- **OpenCV** (`cv2`): Video processing and frame extraction
- **EasyOCR**: Primary text detection engine
- **Tesseract** (`pytesseract`): Backup OCR engine
- **Whisper**: Speech-to-text transcription
- **MoviePy**: Video/audio manipulation
- **Transformers**: Sentiment analysis models
- **PyTorch**: Machine learning framework
- **NumPy**: Numerical computations

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB+ for models and temporary files

## Troubleshooting

### Common Issues

#### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### Tesseract Not Found
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

#### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
- Reduce Whisper model size in config
- Lower video resolution limits
- Process shorter video segments

### Performance Issues

- **Slow processing**: Use smaller models, enable GPU, reduce resolution
- **High memory usage**: Use "tiny" or "base" Whisper models
- **Poor accuracy**: Use larger models, adjust confidence thresholds

## Roadmap

### Planned Features

- [ ] Support for additional languages (Spanish, French, German)
- [ ] Real-time streaming analysis
- [ ] REST API service wrapper
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Custom model training pipelines
- [ ] Advanced filtering rules engine

### Version History

- **v1.0.0**: Initial release with basic analysis pipeline
- **v1.1.0**: Added multi-language support
- **v1.2.0**: Improved false positive filtering
- **Current**: Enhanced configuration management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For technical support and questions:

- **Issues**: [GitHub Issues](https://github.com/your-company/video-content-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-company/video-content-analyzer/discussions)
- **Email**: support@yourcompany.com

## Acknowledgments

- OpenAI for the Whisper speech-to-text model
- The EasyOCR team for excellent OCR capabilities
- Hugging Face for transformer models and infrastructure
- The open-source community for various supporting libraries

---

**Note**: This tool is designed for content moderation assistance. Always combine automated analysis with human review for critical applications. The accuracy and effectiveness may vary based on content type, language, and context.