const EMOTION_META = {
    angry: { icon: '↗', label: 'Angry' },
    calm: { icon: '≈', label: 'Calm' },
    disgust: { icon: '◌', label: 'Disgust' },
    fear: { icon: '!', label: 'Fear' },
    happy: { icon: '✦', label: 'Happy' },
    neutral: { icon: '—', label: 'Neutral' },
    surprise: { icon: '✧', label: 'Surprise' },
    sad: { icon: '↓', label: 'Sad' },
};

const EMOTION_COLORS = {
    angry: '#d65d6b', calm: '#18a894', disgust: '#8c72d6', fear: '#5b8def',
    happy: '#d6a23e', neutral: '#82909e', surprise: '#d47d42', sad: '#5f78ac',
};

const state = { file: null, previewUrl: null, chart: null, probabilities: null };

const $ = (selector) => document.querySelector(selector);

function showToast(message, kind = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast ${kind}`;
    toast.setAttribute('role', 'status');
    toast.textContent = message;
    document.body.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4800);
}

function setTheme(theme) {
    const dark = theme === 'dark';
    document.body.classList.toggle('dark-theme', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light');
    const icon = $('.theme-icon');
    const button = $('#themeToggle');
    if (icon) icon.textContent = dark ? '☼' : '◐';
    if (button) button.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    if (state.probabilities) renderChart(state.probabilities);
}

function drawWaveform(buffer) {
    const canvas = $('#waveformCanvas');
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(canvas.clientWidth, 320);
    const height = 82;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent').trim();
    context.globalAlpha = 0.85;
    context.lineWidth = 1.35;
    context.beginPath();
    const data = buffer.getChannelData(0);
    const step = Math.max(1, Math.ceil(data.length / width));
    const middle = height / 2;
    for (let x = 0; x < width; x += 1) {
        let min = 1;
        let max = -1;
        for (let offset = 0; offset < step; offset += 1) {
            const sample = data[(x * step) + offset];
            if (sample === undefined) break;
            min = Math.min(min, sample);
            max = Math.max(max, sample);
        }
        context.moveTo(x, middle + (min * middle * 0.82));
        context.lineTo(x, middle + (max * middle * 0.82));
    }
    context.stroke();
}

async function previewAudio(file) {
    const audio = $('#audioPreview');
    const canvas = $('#waveformCanvas');
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);
    audio.src = state.previewUrl;
    audio.classList.remove('hidden');
    canvas.classList.remove('hidden');
    try {
        const context = new (window.AudioContext || window.webkitAudioContext)();
        const buffer = await context.decodeAudioData(await file.arrayBuffer());
        drawWaveform(buffer);
        await context.close();
    } catch (_error) {
        showToast('The audio preview is unavailable, but you can still try prediction.');
    }
}

function selectFile(file) {
    if (!file) return;
    const allowed = ['.wav', '.mp3', '.flac', '.ogg'];
    const extension = `.${file.name.split('.').pop().toLowerCase()}`;
    if (!allowed.includes(extension)) {
        showToast('Please choose a WAV, MP3, FLAC, or OGG file.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast('The file is larger than the 10 MB upload limit.');
        return;
    }
    state.file = file;
    const prompt = $('#filePrompt');
    const meta = $('#fileMeta');
    if (prompt) prompt.textContent = file.name;
    if (meta) meta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to analyze`;
    $('#dropZone').classList.add('has-file');
    previewAudio(file);
    $('#predictButton').disabled = false;
}

function renderChart(probabilities) {
    const canvas = $('#probabilitiesChart');
    if (!canvas || !window.Chart) return;
    if (state.chart) state.chart.destroy();
    const dark = document.body.classList.contains('dark-theme');
    const textColor = getComputedStyle(document.body).getPropertyValue('--muted').trim();
    const labels = Object.keys(probabilities);
    state.chart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: labels.map((label) => Number((probabilities[label] * 100).toFixed(2))),
                backgroundColor: labels.map((label) => EMOTION_COLORS[label] || '#82909e'),
                borderRadius: 6,
                borderSkipped: false,
                barThickness: 15,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 450 },
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: (item) => ` ${item.raw}%` } } },
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: dark ? '#263541' : '#edf0f3' }, ticks: { color: textColor, callback: (value) => `${value}%`, font: { family: 'DM Mono' } } },
                x: { grid: { display: false }, ticks: { color: textColor, font: { family: 'DM Mono', size: 10 } } },
            },
        },
    });
}

function updateResult(emotion, confidence, probabilities, uncertain, elapsed) {
    const meta = EMOTION_META[emotion] || { icon: '·', label: emotion || 'Unknown' };
    $('#emoji').textContent = meta.icon;
    $('#caption').textContent = meta.label;
    $('#confidence').textContent = `Confidence ${(confidence * 100).toFixed(2)}%${uncertain ? ' · low confidence' : ''}`;
    $('#resultStatus').textContent = uncertain ? 'REVIEW SIGNAL' : 'ANALYSIS COMPLETE';
    $('#resultStatus').className = `result-status ${uncertain ? 'warning' : 'ready'}`;
    $('#featureExtractionTime').textContent = `End-to-end processing · ${elapsed}s`;
    $('#predictionTime').textContent = uncertain ? 'The model is less certain; try a clearer speech sample.' : 'Signal quality looks suitable for this prediction.';
    state.probabilities = probabilities;
    renderChart(probabilities);
    $('#predictionDetails').classList.remove('hidden');
}

async function predictEmotion() {
    if (!state.file) {
        showToast('Choose an audio recording before starting analysis.');
        return;
    }
    const button = $('#predictButton');
    const progress = $('#progressIndicator');
    button.disabled = true;
    $('.button-label').textContent = 'Analyzing signal';
    $('#emoji').textContent = '…';
    $('#caption').textContent = 'Reading the voice';
    $('#confidence').textContent = '';
    $('#predictionDetails').classList.add('hidden');
    progress.classList.remove('hidden');
    $('#progressText').textContent = 'Extracting Wav2Vec2 features…';
    const started = performance.now();
    try {
        const response = await fetch('/upload', { method: 'POST', body: (() => { const form = new FormData(); form.append('file', state.file); return form; })() });
        const body = await response.json().catch(() => ({}));
        if (!response.ok || body.status === 'error') throw new Error(body.message || `Request failed (${response.status})`);
        const result = body.data;
        updateResult(result.predicted_emotion, result.confidence, result.probabilities, result.is_uncertain, ((performance.now() - started) / 1000).toFixed(2));
        await updateHistory();
    } catch (error) {
        $('#emoji').textContent = '×';
        $('#caption').textContent = 'Analysis unavailable';
        $('#confidence').textContent = '';
        showToast(error.message || 'Prediction failed. Please try again.');
    } finally {
        button.disabled = false;
        $('.button-label').textContent = 'Analyze emotion';
        progress.classList.add('hidden');
    }
}

async function updateHistory() {
    const body = $('#predictionsBody');
    try {
        const response = await fetch('/predictions');
        const payload = await response.json();
        body.replaceChildren();
        const predictions = (payload.data || []).slice(0, 10);
        if (!predictions.length) {
            body.innerHTML = '<tr><td colspan="4" class="empty-history">No analyses yet — your recent results will appear here.</td></tr>';
            return;
        }
        predictions.forEach((prediction) => {
            const row = document.createElement('tr');
            const filename = document.createElement('td');
            const emotion = document.createElement('td');
            const confidence = document.createElement('td');
            const timestamp = document.createElement('td');
            filename.textContent = prediction.filename || 'Unknown file';
            emotion.textContent = `${prediction.predicted_emotion || 'Unknown'}${prediction.is_uncertain ? ' · review' : ''}`;
            confidence.textContent = prediction.confidence == null ? '—' : `${(prediction.confidence * 100).toFixed(2)}%`;
            timestamp.textContent = prediction.timestamp || '—';
            row.append(filename, emotion, confidence, timestamp);
            body.appendChild(row);
        });
    } catch (_error) {
        body.innerHTML = '<tr><td colspan="4" class="empty-history">History is temporarily unavailable.</td></tr>';
    }
}

function setup() {
    const saved = localStorage.getItem('theme');
    setTheme(saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
    $('#themeToggle').addEventListener('click', () => setTheme(document.body.classList.contains('dark-theme') ? 'light' : 'dark'));
    $('#predictButton').disabled = true;
    $('#predictButton').addEventListener('click', predictEmotion);
    const dropZone = $('#dropZone');
    const input = $('#audioFile');
    dropZone.addEventListener('click', () => input.click());
    dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); } });
    input.addEventListener('change', () => selectFile(input.files[0]));
    ['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragover'); }));
    dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
    window.addEventListener('resize', () => { if (state.file && $('#waveformCanvas') && !$('#waveformCanvas').classList.contains('hidden')) previewAudio(state.file); });
    window.addEventListener('scroll', () => $('.topbar').classList.toggle('is-scrolled', window.scrollY > 5), { passive: true });
    updateHistory();
}

document.addEventListener('DOMContentLoaded', setup);
