// PANAS item categories
const POSITIVE_ITEMS = ['interested', 'excited', 'strong', 'enthusiastic', 'proud', 'alert', 'inspired', 'determined', 'attentive', 'active'];
const NEGATIVE_ITEMS = ['distressed', 'upset', 'guilty', 'scared', 'hostile', 'irritable', 'ashamed', 'nervous', 'jittery', 'afraid'];

// Data storage
let entries = [];
let chart = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadData();
    setDefaultDate();
    setupFormSubmission();
    initializeChart();
    updateChart();
});

// Set today's date as default
function setDefaultDate() {
    const dateInput = document.getElementById('entry-date');
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;
    dateInput.max = today; // Prevent future dates
}

// Load data from localStorage
function loadData() {
    const stored = localStorage.getItem('panasEntries');
    if (stored) {
        entries = JSON.parse(stored);
    }
}

// Save data to localStorage
function saveData() {
    localStorage.setItem('panasEntries', JSON.stringify(entries));
}

// Calculate scores
function calculateScores(formData) {
    let positiveScore = 0;
    let negativeScore = 0;

    POSITIVE_ITEMS.forEach(item => {
        positiveScore += parseInt(formData[item]);
    });

    NEGATIVE_ITEMS.forEach(item => {
        negativeScore += parseInt(formData[item]);
    });

    return { positiveScore, negativeScore };
}

// Handle form submission
function setupFormSubmission() {
    const form = document.getElementById('panas-form');
    form.addEventListener('submit', (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = {};

        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }

        const date = document.getElementById('entry-date').value;
        const scores = calculateScores(data);

        const entry = {
            date: date,
            items: data,
            positiveScore: scores.positiveScore,
            negativeScore: scores.negativeScore,
            period: data.period === 'yes',
            notes: data.notes || ''
        };

        // Check if entry for this date already exists
        const existingIndex = entries.findIndex(e => e.date === date);
        if (existingIndex >= 0) {
            if (confirm('An entry for this date already exists. Do you want to overwrite it?')) {
                entries[existingIndex] = entry;
            } else {
                return;
            }
        } else {
            entries.push(entry);
        }

        // Sort entries by date
        entries.sort((a, b) => new Date(a.date) - new Date(b.date));

        saveData();
        updateChart();
        form.reset();
        setDefaultDate();

        alert('Entry saved successfully!');
    });
}

// Initialize Chart.js
function initializeChart() {
    const ctx = document.getElementById('mood-chart').getContext('2d');

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Positive Affect',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#10b981',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'Negative Affect',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#ef4444',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        afterBody: function(context) {
                            const index = context[0].dataIndex;
                            const entry = entries[index];
                            if (entry && entry.period) {
                                return '\n🔴 Period day';
                            }
                            return '';
                        }
                    }
                },
                annotation: {
                    annotations: {}
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 10,
                    max: 50,
                    title: {
                        display: true,
                        text: 'Affect Score',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    showNotes(entries[index]);
                }
            }
        }
    });
}

// Update chart with current data
function updateChart() {
    if (!chart || entries.length === 0) return;

    const dates = entries.map(e => formatDate(e.date));
    const positiveScores = entries.map(e => e.positiveScore);
    const negativeScores = entries.map(e => e.negativeScore);

    chart.data.labels = dates;
    chart.data.datasets[0].data = positiveScores;
    chart.data.datasets[1].data = negativeScores;

    // Add period shading annotations
    const annotations = {};
    entries.forEach((entry, index) => {
        if (entry.period) {
            annotations[`period-${index}`] = {
                type: 'box',
                xMin: index - 0.5,
                xMax: index + 0.5,
                yMin: 10,
                yMax: 50,
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 0
            };
        }
    });

    chart.options.plugins.annotation.annotations = annotations;
    chart.update();
}

// Format date for display
function formatDate(dateString) {
    const date = new Date(dateString + 'T00:00:00');
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

// Show notes popup
function showNotes(entry) {
    const popup = document.getElementById('notes-display');
    const dateElem = document.getElementById('notes-date');
    const textElem = document.getElementById('notes-text');
    const scoresElem = document.getElementById('notes-scores');

    dateElem.textContent = formatDate(entry.date);
    textElem.textContent = entry.notes || 'No notes for this day.';
    scoresElem.innerHTML = `
        <p style="color: #10b981;">✓ Positive Affect: ${entry.positiveScore}/50</p>
        <p style="color: #ef4444;">✓ Negative Affect: ${entry.negativeScore}/50</p>
        ${entry.period ? '<p style="color: #f59e0b;">🔴 Period day</p>' : ''}
    `;

    popup.style.display = 'block';
}

// Close notes popup
document.addEventListener('click', (e) => {
    const popup = document.getElementById('notes-display');
    if (e.target.classList.contains('close-btn')) {
        popup.style.display = 'none';
    }
    if (e.target === popup) {
        popup.style.display = 'none';
    }
});
