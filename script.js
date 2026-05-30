const webcamElement = document.getElementById('webcam');
const startCameraButton = document.getElementById('startCamera');
const stopCameraButton = document.getElementById('stopCamera');
const startRecordingButton = document.getElementById('startRecording');
const stopRecordingButton = document.getElementById('stopRecording');
const downloadVideoButton = document.getElementById('downloadVideo');
const statusPill = document.getElementById('status');
const recognizedText = document.getElementById('recognizedText');
const translatedText = document.getElementById('translatedText');
const translateButton = document.getElementById('translateButton');
const languageSelect = document.getElementById('languageSelect');
const simulateButton = document.getElementById('simulateProcess');
const voiceSelect = document.getElementById('voiceSelect');
const playAudioButton = document.getElementById('playAudioButton');
const stopAudioButton = document.getElementById('stopAudioButton');
const audioPlayer = document.getElementById('audioPlayer');

const ELEVENLABS_API_KEY = 'sk_e5a40034935a1c992d3b42e098f7b07befa48ec841720182';

let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;

function setStatus(message) {
  statusPill.textContent = message;
}

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcamElement.srcObject = mediaStream;

    startCameraButton.disabled = true;
    stopCameraButton.disabled = false;
    startRecordingButton.disabled = false;
    setStatus('Camera connected');
  } catch (error) {
    console.error('Camera start failed:', error);
    setStatus('Camera unavailable');
  }
}

function stopCamera() {
  if (!mediaStream) return;
  mediaStream.getTracks().forEach((track) => track.stop());
  webcamElement.srcObject = null;
  mediaStream = null;

  startCameraButton.disabled = false;
  stopCameraButton.disabled = true;
  startRecordingButton.disabled = true;
  stopRecordingButton.disabled = true;
  downloadVideoButton.disabled = !recordedBlob;
  setStatus('Camera stopped');
}

function startRecording() {
  if (!mediaStream) return;
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm; codecs=vp9' });

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
    downloadVideoButton.disabled = false;
    uploadVideoButton.disabled = false;
    setStatus('Recording complete');
  };

  mediaRecorder.start();
  startRecordingButton.disabled = true;
  stopRecordingButton.disabled = false;
  downloadVideoButton.disabled = true;
  uploadVideoButton.disabled = true;
function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
  mediaRecorder.stop();
  startRecordingButton.disabled = false;
  stopRecordingButton.disabled = true;
}

function downloadVideo() {
  if (!recordedBlob) return;
  const url = URL.createObjectURL(recordedBlob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = 'asl_recording.webm';
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

async function sendRecordedVideo() {
  if (!recordedBlob) {
    setStatus('No video recorded yet');
    return;
  }

  setStatus('Uploading video to backend...');

  const formData = new FormData();
  formData.append('aslVideo', recordedBlob, 'asl_recording.webm');

  try {
    const response = await fetch('/api/process-video', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    const result = await response.json();
    recognizedText.textContent = result.recognizedText || 'No recognized text returned.';
    translatedText.textContent = result.translatedText || 'No translated text returned.';
    setStatus('Backend processing complete');
  } catch (error) {
    console.error(error);
    setStatus('Backend upload failed');
  }
}

function translateText() {
  const source = recognizedText.textContent.trim();
  if (!source || source === 'Waiting for input...' || source === 'No recognized text returned.') {
    setStatus('No recognized text to translate');
    return;
  }

  const targetLanguage = languageSelect.value;
  setStatus('Translating...');

  // Placeholder translation logic.
  // Replace this with a real translation API call or backend translation endpoint.
  const simulatedTranslation = `(${targetLanguage}) ${source}`;
  translatedText.textContent = simulatedTranslation;
  playAudioButton.disabled = false;
  setStatus('Translation ready');
}

async function synthesizeAndPlayAudio() {
  const textToSpeak = translatedText.textContent.trim();
  if (!textToSpeak || textToSpeak === 'Select a language and translate.') {
    setStatus('No translated text to speak');
    return;
  }

  const voiceId = voiceSelect.value;
  playAudioButton.disabled = true;
  setStatus('Generating audio...');

  try {
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text: textToSpeak,
        model_id: 'eleven_monolingual_v1',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`ElevenLabs API error: ${response.status}`);
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    audioPlayer.src = audioUrl;
    audioPlayer.play();
    stopAudioButton.disabled = false;
    setStatus('Playing audio');
  } catch (error) {
    console.error('Audio synthesis failed:', error);
    setStatus('Audio synthesis failed');
    playAudioButton.disabled = false;
  }
}

function stopAudio() {
  audioPlayer.pause();
  audioPlayer.currentTime = 0;
  stopAudioButton.disabled = true;
  setStatus('Audio stopped');
}

audioPlayer.addEventListener('ended', () => {
  stopAudioButton.disabled = true;
  setStatus('Audio playback complete');
});

function simulateBackendProcessing() {
  recognizedText.textContent = 'Hello, how are you?';
  translatedText.textContent = 'Hola, ¿cómo estás?';
  setStatus('Simulation complete');
}

startCameraButton.addEventListener('click', startCamera);
stopCameraButton.addEventListener('click', stopCamera);
startRecordingButton.addEventListener('click', startRecording);
stopRecordingButton.addEventListener('click', stopRecording);
downloadVideoButton.addEventListener('click', downloadVideo);
translateButton.addEventListener('click', translateText);
simulateButton.addEventListener('click', simulateBackendProcessing);

const uploadVideoButton = document.getElementById('uploadVideo');
uploadVideoButton.addEventListener('click', sendRecordedVideo);
playAudioButton.addEventListener('click', synthesizeAndPlayAudio);
stopAudioButton.addEventListener('click', stopAudio);

window.addEventListener('beforeunload', () => {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
});
