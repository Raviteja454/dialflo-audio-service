
# Dialflo Audio Analysis Service

A FastAPI-based audio analysis service that processes uploaded audio, evaluates audio quality, extracts acoustic features, and returns gender and age-bracket predictions.

## Tech Stack

- Python 3.11
- FastAPI
- NumPy
- Librosa
- FFmpeg
- Pydantic
- Pytest

## Setup

### 1. Clone

```bash
git clone https://github.com/Raviteja454/dialflo-audio-service.git
cd dialflo-audio-service
````

### 2. Create Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install FFmpeg

Windows:

```powershell
winget install Gyan.FFmpeg.Shared
```

Verify:

```powershell
ffmpeg -version
```

If FFmpeg is installed in a custom location, configure:

```powershell
$env:FFMPEG_PATH="C:\path\to\ffmpeg.exe"
```

## Run Tests

```powershell
python -m pytest -v
```

## Run Application

```powershell
python -m uvicorn app.main:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Health Check

```powershell
curl.exe http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "service": "dialflo-audio-service"
}
```

## Analyze Audio

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze" -F "file=@.\tests\sample.wav"
```

Example response:

```json
{
  "contact_id": "c243f522-7197-4664-950d-5e9c310315e4",
  "gender": {
    "prediction": "female",
    "confidence": 0.61
  },
  "age_bracket": {
    "prediction": "46-60",
    "confidence": 0.60
  },
  "processing_ms": 139.76,
  "audio_quality": "good"
}
```

## Postman

Create:

```text
POST http://127.0.0.1:8000/analyze
```

Select:

```text
Body → form-data
```

Add:

```text
Key: file
Type: File
Value: Select an audio file
```

## Audio Processing

```text
Upload
  ↓
Validation
  ↓
FFmpeg Decode
  ↓
Mono 16 kHz Audio
  ↓
Quality Analysis
  ↓
Feature Extraction
  ↓
Classifier
  ↓
JSON Response
```

Extracted features include:

* RMS
* Zero Crossing Rate
* Spectral Centroid
* Spectral Bandwidth

## Classifier

The current `HeuristicAudioClassifier` is a lightweight baseline using acoustic features.

It is **not a trained demographic classification model** and should not be considered production-accurate for gender or age prediction.

The classifier is implemented behind an abstraction so it can later be replaced with a trained ML model without changing the API layer.

## Configuration

Default configuration:

```text
Maximum audio size: 15 MB
Minimum duration: 1 second
Maximum duration: 30 seconds
Sample rate: 16000 Hz
Confidence threshold: 0.60
```

## Project Structure

```text
app/
├── api/
├── audio/
├── inference/
├── config.py
├── main.py
└── schemas.py

tests/
├── test_analyze.py
├── test_classifier.py
├── test_features.py
├── test_health.py
└── test_quality.py
```

## Future Improvements

* Replace heuristic classifier with a trained ML model
* Model versioning and confidence calibration
* Authentication and rate limiting
* Monitoring and metrics
* CI/CD
* Docker deployment
* Additional test coverage
