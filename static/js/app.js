const stream = document.getElementById('video-stream');
const empty = document.getElementById('camera-empty');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const captureButton = document.getElementById('capture-button');
const toast = document.getElementById('toast');
const confidenceFill = document.querySelector('.confidence-bar i');
const resultBox = document.querySelector('.result-box');

function notify(message) { toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2800); }
async function request(url, options = {}) { const response = await fetch(url, options); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed'); return data; }
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
function updateStatus(status) {
  const active = status.camera_active;
  const found = status.aquafina_detected;
  const state = document.getElementById('detection-state');
  const live = document.getElementById('live-dot');
  startButton.disabled = active || !status.model_ready;
  stopButton.disabled = !active;
  captureButton.disabled = !active;
  live.classList.toggle('active', active); live.innerHTML = `<i></i> ${active ? 'LIVE' : 'OFFLINE'}`;
  setText('model-status', status.model_ready ? 'Ready' : 'Unavailable'); setText('camera-status', active ? 'Active' : 'Off'); setText('scan-status', active ? 'Scanning' : 'Idle');
  document.getElementById('model-status').classList.toggle('off', !status.model_ready); document.getElementById('camera-status').classList.toggle('off', !active); document.getElementById('scan-status').classList.toggle('off', !active);
  const confidence = found ? status.confidence : 0;
  setText('confidence', found ? `${(confidence * 100).toFixed(1)}%` : '—'); setText('detections', active ? status.detection_count : '0');
  confidenceFill.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  confidenceFill.classList.toggle('accepted', found);
  state.classList.toggle('detected', found); updateResult(found, active);
  if (found) { setText('status-message', 'Authentic Aquafina visual signature recognized.'); state.querySelector('h3').textContent = 'Aquafina Detected'; }
  else if (active) { setText('status-message', 'No Aquafina currently detected.'); state.querySelector('h3').textContent = 'Scanning...'; }
  else { setText('status-message', status.error || 'Start the camera when you are ready.'); state.querySelector('h3').textContent = 'Ready to Scan'; }
}
async function refreshStatus() { try { updateStatus(await request('/api/status')); } catch (_) {} }
startButton.addEventListener('click', async () => { try { await request('/api/camera/start', { method: 'POST' }); stream.src = `/video_feed?ts=${Date.now()}`; stream.style.display = 'block'; empty.style.display = 'none'; await refreshStatus(); } catch (error) { notify(error.message); } });
stopButton.addEventListener('click', async () => { try { await request('/api/camera/stop', { method: 'POST' }); stream.removeAttribute('src'); stream.style.display = 'none'; empty.style.display = 'grid'; await refreshStatus(); } catch (error) { notify(error.message); } });
captureButton.addEventListener('click', async () => { try { const data = await request('/api/capture', { method: 'POST' }); notify(`Captured ${data.filename}`); } catch (error) { notify(error.message); } });
refreshStatus(); setInterval(refreshStatus, 1000);
