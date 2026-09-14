const MODEL_URL = '/static/models/best.onnx';
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.75;
const NMS_IOU_THRESHOLD = 0.45;
const CLASS_NAME = 'Aquafina';
const INFERENCE_INTERVAL_MS = 75;

const streamElement = document.getElementById('video-stream');
const empty = document.getElementById('camera-empty');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const captureButton = document.getElementById('capture-button');
const toast = document.getElementById('toast');
const confidenceFill = document.querySelector('.confidence-bar i');
const resultBox = document.querySelector('.result-box');
const cameraStage = document.getElementById('camera-stage');
const video = document.createElement('video');
const detectionCanvas = document.createElement('canvas');
video.id = 'camera-video';
video.autoplay = true;
video.muted = true;
video.playsInline = true;
detectionCanvas.id = 'detection-canvas';
detectionCanvas.setAttribute('aria-hidden', 'true');
streamElement.replaceWith(video);
cameraStage.appendChild(detectionCanvas);

let session = null;
let mediaStream = null;
let inferenceRunning = false;
let inferenceBusy = false;
let lastDetections = [];
let modelState = 'Loading';
let runtimePromise = null;

function notify(message) { toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2800); }
function setText(id, value) { document.getElementById(id).textContent = value; }
function updateResult(found, active) {
  const title = resultBox.querySelector('strong');
  const detail = resultBox.querySelector('span');
  resultBox.classList.toggle('accepted', found);
  resultBox.classList.remove('rejected');
  if (found) { title.textContent = 'AQUAFINA — ACCEPTED'; detail.textContent = 'Visual signature recognized.'; }
  else if (active) { title.textContent = 'Waiting for detection...'; detail.textContent = 'Point your camera at a bottle.'; }
  else { title.textContent = 'Waiting for detection...'; detail.textContent = 'Start the camera to begin scanning.'; }
}
function updateStatus(active, error = null) {
  const found = lastDetections.length > 0;
  const state = document.getElementById('detection-state');
  const live = document.getElementById('live-dot');
  startButton.disabled = active || modelState === 'Loading';
  stopButton.disabled = !active;
  captureButton.disabled = !active;
  live.classList.toggle('active', active); live.innerHTML = `<i></i> ${active ? 'LIVE' : 'OFFLINE'}`;
  setText('model-status', modelState); setText('camera-status', active ? 'Active' : 'Off'); setText('scan-status', active ? 'Scanning' : 'Idle');
  document.getElementById('model-status').classList.toggle('off', modelState === 'Error'); document.getElementById('camera-status').classList.toggle('off', !active); document.getElementById('scan-status').classList.toggle('off', !active);
  const confidence = found ? lastDetections[0].confidence : 0;
  setText('confidence', found ? `${(confidence * 100).toFixed(1)}%` : '—'); setText('detections', active ? String(lastDetections.length) : '0');
  confidenceFill.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  confidenceFill.classList.toggle('accepted', found);
  state.classList.toggle('detected', found); updateResult(found, active);
  if (found) { setText('status-message', 'Authentic Aquafina visual signature recognized.'); state.querySelector('h3').textContent = 'Aquafina Detected'; }
  else if (active) { setText('status-message', 'No Aquafina currently detected.'); state.querySelector('h3').textContent = 'Scanning...'; }
  else { setText('status-message', error || 'Start the camera when you are ready.'); state.querySelector('h3').textContent = 'Ready to Scan'; }
}

async function loadModel() {
  if (session) return session;
  modelState = 'Loading'; updateStatus(false);
  if (!window.ort) {
    runtimePromise ||= new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js';
      script.onload = resolve;
      script.onerror = () => reject(new Error('ONNX Runtime Web could not be loaded.'));
      document.head.appendChild(script);
    });
    await runtimePromise;
  }
  if (!window.ort) throw new Error('ONNX Runtime Web did not load.');
  if (navigator.gpu) {
    try { session = await window.ort.InferenceSession.create(MODEL_URL, { executionProviders: ['webgpu'] }); }
    catch (_) { session = null; }
  }
  if (!session) session = await window.ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  modelState = 'Ready'; updateStatus(false);
  return session;
}

function preprocessFrame() {
  const originalWidth = video.videoWidth;
  const originalHeight = video.videoHeight;
  const scale = Math.min(INPUT_SIZE / originalWidth, INPUT_SIZE / originalHeight);
  const resizedWidth = Math.max(1, Math.round(originalWidth * scale));
  const resizedHeight = Math.max(1, Math.round(originalHeight * scale));
  const paddingX = Math.floor((INPUT_SIZE - resizedWidth) / 2);
  const paddingY = Math.floor((INPUT_SIZE - resizedHeight) / 2);
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = resizedWidth; sourceCanvas.height = resizedHeight;
  const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
  sourceContext.drawImage(video, 0, 0, resizedWidth, resizedHeight);
  const source = sourceContext.getImageData(0, 0, resizedWidth, resizedHeight).data;
  // Canvas pixels are RGB(A); write normalized channels directly in CHW order.
  const data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  data.fill(114 / 255);
  for (let y = 0; y < resizedHeight; y += 1) {
    for (let x = 0; x < resizedWidth; x += 1) {
      const sourceIndex = (y * resizedWidth + x) * 4;
      const targetIndex = (y + paddingY) * INPUT_SIZE + x + paddingX;
      data[targetIndex] = source[sourceIndex] / 255;
      data[INPUT_SIZE * INPUT_SIZE + targetIndex] = source[sourceIndex + 1] / 255;
      data[2 * INPUT_SIZE * INPUT_SIZE + targetIndex] = source[sourceIndex + 2] / 255;
    }
  }
  return { tensor: new window.ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]), scale, paddingX, paddingY, originalWidth, originalHeight };
}

function iou(first, second) {
  const left = Math.max(first.x1, second.x1); const top = Math.max(first.y1, second.y1);
  const right = Math.min(first.x2, second.x2); const bottom = Math.min(first.y2, second.y2);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = first.area + second.area - intersection;
  return union > 0 ? intersection / union : 0;
}
function nms(detections) {
  const candidates = detections.slice().sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  while (candidates.length) {
    const current = candidates.shift();
    kept.push(current);
    for (let index = candidates.length - 1; index >= 0; index -= 1) {
      if (iou(current, candidates[index]) > NMS_IOU_THRESHOLD) candidates.splice(index, 1);
    }
  }
  return kept;
}
function postprocessDetections(output, metadata) {
  if (output.dims.join(',') !== '1,5,8400') throw new Error(`Unexpected output shape: ${output.dims.join(',')}`);
  const values = output.data; const detections = [];
  // output0 is [1, 5, 8400], so each channel is a contiguous 8400-value block.
  for (let index = 0; index < 8400; index += 1) {
    const confidence = values[4 * 8400 + index];
    if (confidence < CONFIDENCE_THRESHOLD) continue;
    const centerX = values[index]; const centerY = values[8400 + index];
    const width = values[2 * 8400 + index]; const height = values[3 * 8400 + index];
    const x1 = Math.max(0, Math.min(metadata.originalWidth - 1, (centerX - width / 2 - metadata.paddingX) / metadata.scale));
    const y1 = Math.max(0, Math.min(metadata.originalHeight - 1, (centerY - height / 2 - metadata.paddingY) / metadata.scale));
    const x2 = Math.max(0, Math.min(metadata.originalWidth - 1, (centerX + width / 2 - metadata.paddingX) / metadata.scale));
    const y2 = Math.max(0, Math.min(metadata.originalHeight - 1, (centerY + height / 2 - metadata.paddingY) / metadata.scale));
    if (x2 <= x1 || y2 <= y1) continue;
    detections.push({ label: CLASS_NAME, classId: 0, confidence, x1, y1, x2, y2, area: (x2 - x1) * (y2 - y1) });
  }
  return nms(detections);
}
function resizeDetectionCanvas() {
  detectionCanvas.width = video.videoWidth || 1; detectionCanvas.height = video.videoHeight || 1;
  detectionCanvas.style.width = `${video.clientWidth}px`; detectionCanvas.style.height = `${video.clientHeight}px`;
}
function drawDetections(detections) {
  const context = detectionCanvas.getContext('2d'); context.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
  const scaleX = detectionCanvas.width / video.videoWidth; const scaleY = detectionCanvas.height / video.videoHeight;
  context.lineWidth = Math.max(2, detectionCanvas.width / 320); context.font = `${Math.max(12, detectionCanvas.width / 42)}px Manrope, sans-serif`;
  detections.forEach((detection) => { const x = detection.x1 * scaleX; const y = detection.y1 * scaleY; const width = (detection.x2 - detection.x1) * scaleX; const height = (detection.y2 - detection.y1) * scaleY; const label = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`; context.strokeStyle = '#16A34A'; context.fillStyle = '#16A34A'; context.strokeRect(x, y, width, height); const labelWidth = context.measureText(label).width + 12; context.fillRect(x, Math.max(0, y - 25), labelWidth, 25); context.fillStyle = '#fff'; context.fillText(label, x + 6, Math.max(17, y - 7)); });
}
async function runInference() {
  if (!inferenceRunning || inferenceBusy || video.readyState < 2) return;
  inferenceBusy = true;
  try { const prepared = preprocessFrame(); const results = await session.run({ images: prepared.tensor }); lastDetections = postprocessDetections(results.output0, prepared); drawDetections(lastDetections); updateStatus(true); }
  catch (error) { notify(`Inference failed: ${error.message}`); stopCamera(); }
  finally { inferenceBusy = false; }
}
function inferenceLoop() { if (!inferenceRunning) return; runInference().finally(() => { if (inferenceRunning) setTimeout(inferenceLoop, INFERENCE_INTERVAL_MS); }); }
async function startCamera() {
  if (inferenceRunning) return;
  try { await loadModel(); mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); video.srcObject = mediaStream; await video.play(); empty.style.display = 'none'; video.style.display = 'block'; resizeDetectionCanvas(); inferenceRunning = true; updateStatus(true); inferenceLoop(); }
  catch (error) { modelState = session ? 'Ready' : 'Error'; const message = error.name === 'NotAllowedError' ? 'Camera permission was denied.' : error.name === 'NotFoundError' ? 'No camera was found.' : error.name === 'NotReadableError' ? 'The camera is already in use.' : error.message || 'Unable to start the camera.'; notify(message); updateStatus(false, message); stopCamera(); }
}
function stopCamera() { inferenceRunning = false; inferenceBusy = false; if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; video.srcObject = null; video.style.display = 'none'; empty.style.display = 'grid'; lastDetections = []; detectionCanvas.getContext('2d').clearRect(0, 0, detectionCanvas.width, detectionCanvas.height); updateStatus(false); }
function captureFrame() { const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight; const context = canvas.getContext('2d'); context.drawImage(video, 0, 0); context.drawImage(detectionCanvas, 0, 0); const link = document.createElement('a'); link.download = `aquafina-capture-${new Date().toISOString().replace(/[:.]/g, '-')}.png`; link.href = canvas.toDataURL('image/png'); link.click(); notify(`Captured ${link.download}`); }

video.addEventListener('loadedmetadata', resizeDetectionCanvas); window.addEventListener('resize', resizeDetectionCanvas);
startButton.addEventListener('click', startCamera); stopButton.addEventListener('click', stopCamera); captureButton.addEventListener('click', captureFrame);
if (!navigator.mediaDevices?.getUserMedia) { modelState = 'Error'; updateStatus(false, 'Camera access requires a secure browser context.'); }
else { loadModel().catch((error) => { modelState = 'Error'; updateStatus(false, `Model unavailable: ${error.message}`); notify(`Model loading failed: ${error.message}`); }); }
