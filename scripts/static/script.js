console.log('Loading script.js');

const emojiMap = {
    'angry': '😤',
    'calm': '😌',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😄',
    'neutral': '😐',
    'surprise': '😲',
    'sad': '😢',
    'unknown': '❓'
};

const captionMap = {
    'angry': 'Feeling Angry!',
    'calm': 'Calm & Relaxed.',
    'disgust': 'Yuck, Disgusting!',
    'fear': 'Scared Stiff!',
    'happy': 'Over the Moon!',
    'neutral': 'Keeping Neutral.',
    'surprise': 'Surprised!',
    'sad': 'Down in the Dumps.',
    'unknown': 'Unknown Emotion.'
};

function showError(message, suggestion = '') {
    console.error('Error:', message);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `${message}${suggestion ? `<br><small>${suggestion}</small>` : ''}`;
    document.body.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 5000);
}

function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    body.classList.toggle('dark-theme');
    const isDark = body.classList.contains('dark-theme');
    themeToggle.querySelector('.theme-icon').textContent = isDark ? '☀️' : '🌙';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

function drawWaveform(audioBuffer) {
    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = document.body.classList.contains('dark-theme') ? '#37474f' : '#e3f2fd';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = '#0288d1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, amp);

    for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;
        for (let j = 0; j < step; j++) {
            const datum = data[i * step + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }
        ctx.lineTo(i, (1 + min) * amp);
        ctx.lineTo(i, (1 + max) * amp);
    }

    ctx.stroke();
}

function visualizeAudio(file) {
    const audioPreview = document.getElementById('audioPreview');
    const waveformCanvas = document.getElementById('waveformCanvas');
    audioPreview.src = URL.createObjectURL(file);
    audioPreview.style.display = 'block';
    waveformCanvas.style.display = 'block';

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const reader = new FileReader();
    reader.onload = function(e) {
        audioContext.decodeAudioData(e.target.result, (buffer) => {
            drawWaveform(buffer);
        }, (err) => {
            showError('Failed to decode audio for visualization.', 'Ensure the audio file is not corrupted.');
        });
    };
    reader.onerror = () => {
        showError('Failed to read audio file for visualization.', 'Try a different audio file.');
    };
    reader.readAsArrayBuffer(file);
}

function displayProbabilitiesChart(probabilities) {
    const ctx = document.getElementById('probabilitiesChart');
    if (!ctx || !window.Chart) {
        console.error('Chart.js not loaded or canvas not found');
        showError('Unable to display probability chart.', 'Ensure Chart.js is loaded.');
        return;
    }
    const chartContext = ctx.getContext('2d');
    if (window.probabilityChart) {
        window.probabilityChart.destroy();
    }

    const emotionColors = {
        'angry': '#ef5350',
        'calm': '#26a69a',
        'disgust': '#ab47bc',
        'fear': '#42a5f5',
        'happy': '#66bb6a',
        'neutral': '#90a4ae',
        'surprise': '#ffca28',
        'sad': '#ff8f00'
    };

    const labels = Object.keys(probabilities);
    const bgColors = labels.map(l => emotionColors[l] || '#d3d3d3');

    window.probabilityChart = new Chart(chartContext, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Emotion Probabilities',
                data: Object.values(probabilities).map(v => (v * 100).toFixed(2)),
                backgroundColor: bgColors,
                borderColor: bgColors,
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: document.body.classList.contains('dark-theme') ? '#546e7a' : '#e0e0e0' },
                    ticks: { color: document.body.classList.contains('dark-theme') ? '#eceff1' : '#212121' },
                    title: {
                        display: true,
                        text: 'Probability (%)',
                        color: document.body.classList.contains('dark-theme') ? '#eceff1' : '#212121',
                        font: { size: 14, weight: '500' }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: document.body.classList.contains('dark-theme') ? '#eceff1' : '#212121' },
                    title: {
                        display: true,
                        text: 'Emotions',
                        color: document.body.classList.contains('dark-theme') ? '#eceff1' : '#212121',
                        font: { size: 14, weight: '500' }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: document.body.classList.contains('dark-theme') ? '#eceff1' : '#212121',
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.raw}%`
                    }
                }
            }
        }
    });
}

async function predictEmotion() {
    console.log('predictEmotion called');
    const fileInput = document.getElementById('audioFile');
    const predictButton = document.getElementById('predictButton');
    const emojiDiv = document.getElementById('emoji');
    const captionP = document.getElementById('caption');
    const confidenceP = document.getElementById('confidence');
    const progressIndicator = document.getElementById('progressIndicator');
    const progressText = document.getElementById('progressText');
    const predictionDetails = document.getElementById('predictionDetails');
    const featureExtractionTimeP = document.getElementById('featureExtractionTime');
    const predictionTimeP = document.getElementById('predictionTime');

    if (!fileInput.files.length) {
        showError('Please select an audio file.', 'Choose a .wav, .mp3, or .flac file to proceed.');
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = 'Predicting...';
    emojiDiv.textContent = '⏳';
    captionP.textContent = 'Processing audio...';
    confidenceP.textContent = '';
    progressIndicator.classList.remove('hidden');
    predictionDetails.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        progressText.textContent = 'Uploading & extracting features...';
        const startTime = performance.now();

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const endTime = performance.now();
        const totalTime = ((endTime - startTime) / 1000).toFixed(2);

        console.log('Fetch response status:', response.status);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Fetch response data:', data);
        if (data.status === 'error') {
            throw new Error(data.message);
        }

        const { predicted_emotion, probabilities, confidence, is_uncertain } = data.data;

        emojiDiv.textContent = emojiMap[predicted_emotion] || '❓';

        if (is_uncertain) {
            captionP.textContent = `Possibly ${captionMap[predicted_emotion] || 'Unknown'} (Low confidence)`;
            captionP.style.opacity = '0.7';
        } else {
            captionP.textContent = captionMap[predicted_emotion] || 'Unknown emotion';
            captionP.style.opacity = '1';
        }

        confidenceP.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%${is_uncertain ? ' ⚠️' : ''}`;
        displayProbabilitiesChart(probabilities);

        featureExtractionTimeP.textContent = `Total processing time: ${totalTime} seconds`;
        predictionTimeP.textContent = is_uncertain
            ? '⚠️ Low confidence - the audio may not contain clear emotional speech.'
            : '';
        predictionDetails.classList.remove('hidden');

        updatePreviousPredictions();
    } catch (error) {
        console.error('Fetch error:', error);
        showError(error.message, 'Please try again or check your audio file.');
        emojiDiv.textContent = '❌';
        captionP.textContent = 'Prediction failed';
        confidenceP.textContent = `Error: ${error.message}`;
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = 'Upload & Predict';
        progressIndicator.classList.add('hidden');
    }
}

function updatePreviousPredictions() {
    console.log('updatePreviousPredictions called');
    const tableBody = document.getElementById('predictionsBody');
    fetch('/predictions')
        .then(response => {
            console.log('Predictions fetch status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Predictions data:', data);
            tableBody.innerHTML = '';
            if (data.status === 'error') {
                throw new Error(data.message);
            }
            // FIX: data.data is already newest-first from the server.
            // Do NOT reverse again. Take the first 10.
            data.data.slice(0, 10).forEach(pred => {
                const row = document.createElement('tr');
                // FIX: Use 'filename' field (matches app.py save_prediction).
                // Fall back to 'audio_file' or 'file' for old prediction entries.
                const fileName = pred.filename || pred.audio_file || pred.file || 'Unknown';
                const emotion = pred.predicted_emotion || pred.emotion || 'Unknown';
                const conf = pred.confidence != null ? (pred.confidence * 100).toFixed(2) + '%' : 'N/A';
                const uncertain = pred.is_uncertain ? ' ⚠️' : '';
                const timestamp = pred.timestamp || '';

                row.innerHTML = `
                    <td>${fileName}</td>
                    <td>${emotion}${uncertain}</td>
                    <td>${conf}</td>
                    <td>${timestamp}</td>
                `;
                tableBody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Error loading previous predictions:', error);
            tableBody.innerHTML = '<tr><td colspan="4">Failed to load predictions</td></tr>';
        });
}

function setupDragAndDrop() {
    console.log('setupDragAndDrop called');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('audioFile');

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
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            visualizeAudio(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            visualizeAudio(fileInput.files[0]);
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, setting up drag-and-drop, theme, and predictions');
    setupDragAndDrop();
    updatePreviousPredictions();

    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        document.getElementById('themeToggle').querySelector('.theme-icon').textContent = '☀️';
    }

    document.getElementById('themeToggle').addEventListener('click', () => {
        toggleTheme();
        document.body.dispatchEvent(new Event('themeChange'));
    });

    document.getElementById('audioFile').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            predictEmotion();
        }
    });
});
