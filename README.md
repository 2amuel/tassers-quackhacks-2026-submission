# ASL Translator Frontend

This repository contains a frontend scaffold for a sign language translation website.
It is built to capture webcam video, record it locally, and integrate with a backend or ML pipeline for gesture recognition and translation.

## Files

- `index.html` — main frontend page
- `styles.css` — visual styling for the app
- `script.js` — webcam capture, recording, and backend integration hooks

## Key Features

- Start and stop webcam feed
- Record ASL video using browser `MediaRecorder`
- Download recorded video locally
- Upload recorded video to a backend endpoint for processing
- Display recognized text and translated output
- Support for selecting target language
- **Text-to-speech with ElevenLabs API** — select voice and play audio of translated text

## Integration Points

### Media capture and recording
- `startCamera()` opens the webcam
- `startRecording()` starts recording the live feed
- `stopRecording()` finalizes the recording
- `downloadVideo()` downloads the captured video as `webm`

### Backend processing
- `sendRecordedVideo()` uploads the recorded video to `/api/process-video`
- Backend should accept the video, run MediaPipe Holistic or the custom model, and return JSON:
  - `recognizedText`
  - `translatedText`

### Translation
- `translateText()` is currently a placeholder
- Replace this with a real translation API or backend translation logic

### Text-to-Speech (ElevenLabs)
- `synthesizeAndPlayAudio()` sends the translated text to ElevenLabs TTS API
- Uses the selected voice ID from the voice dropdown
- Plays the generated audio directly in the browser via `<audio>` element
- Supported voices: Rachel, Bella, Callum, Chris, Elli, Glinda, Grace
- API key is stored in `ELEVENLABS_API_KEY` constant

**Security Note:** For production, move the API key to a backend environment variable instead of hardcoding it in the frontend.

## Notes

- Browsers typically record using `webm`; your backend can convert to `mp4` if necessary.
- This frontend is intentionally generic so the team can plug in any backend stack: Flask, Node, FastAPI, TensorFlow Serving, Google AI services, etc.

## How to run

Open `index.html` in a browser, or serve the directory with a static file server.

Example using Python:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.
