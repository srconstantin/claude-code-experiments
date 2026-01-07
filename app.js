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
    setupCSVUpload();
    initializeChart();
    updateChart();
    updateDataInfo();
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

// Setup CSV upload
function setupCSVUpload() {
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('csv-upload');

    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        try {
            const csvText = await file.text();
            await importCSVData(csvText);
            fileInput.value = ''; // Reset file input
        } catch (error) {
            console.error('Error reading CSV file:', error);
            alert('Failed to read CSV file. Please try again.');
        }
    });
}

// Import CSV data from text
async function importCSVData(csvText) {
    try {
        const rows = csvText.trim().split('\n');
        console.log(`Found ${rows.length} rows in CSV`);

        let importedCount = 0;
        let skippedCount = 0;

        // Skip header row
        for (let i = 1; i < rows.length; i++) {
            const row = rows[i];
            const cols = parseCSVRow(row);

            // Need at least date, positive, negative
            if (cols.length < 3) {
                console.log(`Skipping row ${i}: insufficient columns`, cols);
                skippedCount++;
                continue;
            }

            const dateStr = cols[0];
            const positiveAffect = cols[1];
            const negativeAffect = cols[2];
            const period = cols.length > 5 ? cols[5] : '';
            const notes = cols.length > 6 ? cols[6] : '';

            // Skip rows with missing required data
            if (!dateStr || !positiveAffect || !negativeAffect) {
                console.log(`Skipping row ${i}: missing required data`, cols);
                skippedCount++;
                continue;
            }

            // Parse date from M/D/YY or M/D/YYYY format to YYYY-MM-DD
            const dateParts = dateStr.split('/');
            if (dateParts.length !== 3) {
                console.log(`Skipping row ${i}: invalid date format`, dateStr);
                skippedCount++;
                continue;
            }

            const month = dateParts[0].padStart(2, '0');
            const day = dateParts[1].padStart(2, '0');
            // Handle both 2-digit and 4-digit years
            let year = dateParts[2];
            if (year.length === 2) {
                year = '20' + year;
            }
            const formattedDate = `${year}-${month}-${day}`;

            const entry = {
                date: formattedDate,
                items: {}, // CSV doesn't have individual item scores
                positiveScore: parseInt(positiveAffect),
                negativeScore: parseInt(negativeAffect),
                period: period && period.toLowerCase().trim() === 'y',
                notes: notes || ''
            };

            // Check if entry for this date already exists
            const existingIndex = entries.findIndex(e => e.date === formattedDate);
            if (existingIndex >= 0) {
                // Keep the most recent entry (last one in CSV)
                entries[existingIndex] = entry;
            } else {
                entries.push(entry);
            }
            importedCount++;
        }

        // Sort entries by date
        entries.sort((a, b) => new Date(a.date) - new Date(b.date));

        saveData();
        console.log(`Successfully imported ${importedCount} entries from CSV (${skippedCount} skipped)`);
        console.log(`Total entries in storage: ${entries.length}`);

        updateChart();
        updateDataInfo();
        alert(`Successfully imported ${importedCount} entries from CSV!`);
    } catch (error) {
        console.error('Error importing CSV:', error);
        alert('Failed to import CSV data. Check console for details.');
    }
}

// Parse CSV row handling quoted fields
function parseCSVRow(row) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < row.length; i++) {
        const char = row[i];

        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }

    result.push(current.trim());
    return result;
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
    console.log(`Updating chart with ${entries.length} entries`);

    if (!chart) {
        console.log('Chart not initialized yet');
        return;
    }

    if (entries.length === 0) {
        console.log('No entries to display');
        return;
    }

    const dates = entries.map(e => formatDate(e.date));
    const positiveScores = entries.map(e => e.positiveScore);
    const negativeScores = entries.map(e => e.negativeScore);

    console.log(`First entry: ${dates[0]}, Last entry: ${dates[dates.length - 1]}`);

    chart.data.labels = dates;
    chart.data.datasets[0].data = positiveScores;
    chart.data.datasets[1].data = negativeScores;

    // Add period shading annotations
    const annotations = {};
    let periodCount = 0;
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
            periodCount++;
        }
    });

    console.log(`Added ${periodCount} period day annotations`);
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


// Update data info display
function updateDataInfo() {
    const dataInfo = document.getElementById('data-info');
    if (entries.length > 0) {
        const firstDate = formatDate(entries[0].date);
        const lastDate = formatDate(entries[entries.length - 1].date);
        dataInfo.textContent = `Showing ${entries.length} entries from ${firstDate} to ${lastDate}`;
    } else {
        dataInfo.textContent = 'No data to display. Add an entry or check if CSV is loaded.';
    }
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
