// ==================== Constants ====================
const EMOJI_MAP = {
    angry: '😤', calm: '😌', disgust: '🤢', fear: '😨',
    happy: '😄', neutral: '😐', surprise: '😲', sad: '😢'
};

const CAPTION_MAP = {
    angry: 'Feeling Angry', calm: 'Calm & Relaxed', disgust: 'Disgusted',
    fear: 'Scared', happy: 'Happy!', neutral: 'Neutral',
    surprise: 'Surprised!', sad: 'Feeling Sad'
};

const EMOTION_COLORS = {
    angry: '#ef4444', calm: '#10b981', disgust: '#a855f7', fear: '#3b82f6',
    happy: '#22c55e', neutral: '#94a3b8', surprise: '#eab308', sad: '#f97316'
};

// ==================== Toast Notification ====================
function showToast(message, type = 'error') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 4000);
}

// ==================== Theme ====================
function initTheme() {
    const saved = localStorage.getItem('theme');
    // Respect OS dark-mode preference when no saved choice exists
    const preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', saved || preferred);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    // Re-render chart with updated theme colors
    if (lastProbabilities) displayChart(lastProbabilities);
}

// ==================== File Handling ====================
let currentBlobUrl = null;  // track so we can revoke and avoid memory leaks

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function handleFileSelected(file) {
    if (!file) return;

    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const dropZone = document.getElementById('dropZone');
    const audioPreview = document.getElementById('audioPreview');
    const waveformCanvas = document.getElementById('waveformCanvas');

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    dropZone.classList.add('hidden');

    // Revoke any previous blob URL before creating a new one
    if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
    }
    currentBlobUrl = URL.createObjectURL(file);
    audioPreview.src = currentBlobUrl;
    audioPreview.classList.remove('hidden');

    // Draw waveform
    waveformCanvas.classList.remove('hidden');
    drawWaveform(file);
}

function clearFile() {
    const fileInput = document.getElementById('audioFile');
    const fileInfo = document.getElementById('fileInfo');
    const dropZone = document.getElementById('dropZone');
    const audioPreview = document.getElementById('audioPreview');
    const waveformCanvas = document.getElementById('waveformCanvas');

    fileInput.value = '';
    fileInfo.classList.add('hidden');
    dropZone.classList.remove('hidden');
    audioPreview.classList.add('hidden');
    // Revoke and clear the blob URL to free memory
    if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = null;
    }
    audioPreview.src = '';
    waveformCanvas.classList.add('hidden');
}

// ==================== Waveform ====================
function drawWaveform(file) {
    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');

    // Set actual pixel size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    ctx.clearRect(0, 0, width, height);

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const reader = new FileReader();

    reader.onload = function (e) {
        audioContext.decodeAudioData(e.target.result, (buffer) => {
            const data = buffer.getChannelData(0);
            const step = Math.ceil(data.length / width);
            const amp = height / 2;

            // Draw center line
            ctx.strokeStyle = isDark ? 'rgba(148,163,184,0.15)' : 'rgba(0,0,0,0.06)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, amp);
            ctx.lineTo(width, amp);
            ctx.stroke();

            // Draw waveform
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, amp);

            for (let i = 0; i < width; i++) {
                let min = 1.0, max = -1.0;
                for (let j = 0; j < step; j++) {
                    const d = data[i * step + j];
                    if (d !== undefined) {
                        if (d < min) min = d;
                        if (d > max) max = d;
                    }
                }
                ctx.lineTo(i, (1 + min) * amp);
                ctx.lineTo(i, (1 + max) * amp);
            }
            ctx.stroke();

            audioContext.close();
        }).catch(() => {
            audioContext.close(); // always close to avoid AudioContext leak
        });
    };
    reader.readAsArrayBuffer(file);
}

// ==================== Chart ====================
let probabilityChart = null;
let lastProbabilities = null;  // kept so chart can be re-themed on theme toggle

function displayChart(probabilities) {
    lastProbabilities = probabilities;
    const ctx = document.getElementById('probabilitiesChart');
    if (!ctx || !window.Chart) return;

    if (probabilityChart) probabilityChart.destroy();

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities).map(v => (v * 100));
    const colors = labels.map(l => EMOTION_COLORS[l] || '#94a3b8');

    probabilityChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: isDark ? 'rgba(148,163,184,0.1)' : 'rgba(0,0,0,0.06)' },
                    ticks: {
                        color: isDark ? '#94a3b8' : '#64748b',
                        callback: v => v + '%'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: isDark ? '#94a3b8' : '#64748b', font: { size: 11 } }
                }
            }
        }
    });
}

// ==================== Prediction ====================
async function predictEmotion() {
    const fileInput = document.getElementById('audioFile');
    const predictBtn = document.getElementById('predictButton');
    const progress = document.getElementById('progressIndicator');
    const progressText = document.getElementById('progressText');
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');

    if (!fileInput.files.length) {
        showToast('Please select an audio file first.');
        return;
    }

    // UI: Loading state
    predictBtn.disabled = true;
    predictBtn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        Analyzing...`;
    progress.classList.remove('hidden');
    progressText.textContent = 'Uploading and processing audio...';
    resultPlaceholder.classList.add('hidden');
    resultContent.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    const startTime = performance.now();

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const totalTime = ((performance.now() - startTime) / 1000).toFixed(2);

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.message || `Server error (${response.status})`);
        }

        const data = await response.json();
        if (data.status === 'error') throw new Error(data.message);

        const { predicted_emotion, probabilities, confidence, is_uncertain } = data.data;

        // Update emotion badge
        document.getElementById('emoji').textContent = EMOJI_MAP[predicted_emotion] || '?';
        document.getElementById('emotionLabel').textContent = predicted_emotion;
        document.getElementById('caption').textContent =
            is_uncertain ? `Possibly ${CAPTION_MAP[predicted_emotion] || 'Unknown'} (low confidence)`
                         : CAPTION_MAP[predicted_emotion] || predicted_emotion;

        // Update confidence meter
        const confPct = (confidence * 100).toFixed(1);
        document.getElementById('confidenceValue').textContent = confPct + '%';
        const confFill = document.getElementById('confidenceFill');
        confFill.style.width = confPct + '%';
        confFill.className = 'confidence-fill' +
            (confidence >= 0.7 ? '' : confidence >= 0.4 ? ' medium' : ' low');

        const warning = document.getElementById('uncertaintyWarning');
        if (is_uncertain) warning.classList.remove('hidden');
        else warning.classList.add('hidden');

        // Chart
        displayChart(probabilities);

        // Timing
        document.getElementById('processingTime').textContent =
            `Processed in ${totalTime}s`;

        // Show results
        resultContent.classList.remove('hidden');

        // Brief success feedback
        showToast(
            `Detected: ${predicted_emotion} (${confPct}%)`,
            is_uncertain ? 'warning' : 'success'
        );

        // Refresh history
        updateHistory();

    } catch (error) {
        showToast(error.message);
        resultPlaceholder.classList.remove('hidden');
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            Analyze Emotion`;
        progress.classList.add('hidden');
    }
}

// ==================== History ====================
function updateHistory() {
    const tbody = document.getElementById('predictionsBody');

    fetch('/predictions')
        .then(r => r.json())
        .then(data => {
            if (data.status !== 'success' || !data.data.length) {
                tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No predictions yet</td></tr>';
                return;
            }

            tbody.innerHTML = data.data.slice(0, 10).map(pred => {
                const name = pred.filename || 'Unknown';
                const emotion = pred.predicted_emotion || 'Unknown';
                const conf = pred.confidence != null ? (pred.confidence * 100).toFixed(1) : 'N/A';
                const confClass = pred.confidence >= 0.7 ? 'high' : pred.confidence >= 0.4 ? 'medium' : 'low';
                const emoji = EMOJI_MAP[emotion] || '';
                const ts = pred.timestamp || '';
                const uncertain = pred.is_uncertain ? ' *' : '';

                return `<tr>
                    <td title="${name}">${name.length > 25 ? name.slice(0, 22) + '...' : name}</td>
                    <td><span class="emotion-tag">${emoji} ${emotion}${uncertain}</span></td>
                    <td><span class="confidence-tag ${confClass}">${conf}%</span></td>
                    <td>${ts}</td>
                </tr>`;
            }).join('');
        })
        .catch(() => {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">Failed to load history</td></tr>';
        });
}

// ==================== Drag & Drop ====================
function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('audioFile');

    dropZone.addEventListener('click', () => fileInput.click());

    // Keyboard: activate on Enter or Space for accessibility
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileSelected(fileInput.files[0]);
        }
    });

    document.getElementById('clearFile').addEventListener('click', clearFile);
}

// ==================== Init ====================
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupDragAndDrop();
    updateHistory();

    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
});
