# PANAS Mood Tracker

A web application for tracking daily mood using the PANAS (Positive and Negative Affect Schedule) questionnaire.

## Features

- Complete PANAS questionnaire with 20 emotion items
- Period tracking and daily notes
- Interactive line chart showing positive and negative affect over time
- Period days highlighted with light red shading
- Click on data points to view notes
- Automatic CSV import for historical data
- Data stored locally in browser

## How to Run

Since the app needs to load a CSV file, you must run it through a local web server (not by opening the HTML file directly).

### Option 1: Python HTTP Server (Recommended)

If you have Python installed:

```bash
# Navigate to the project directory
cd /home/user/claude-code-experiments

# Python 3
python3 -m http.server 8000

# OR Python 2
python -m SimpleHTTPServer 8000
```

Then open your browser to: `http://localhost:8000`

### Option 2: Node.js HTTP Server

If you have Node.js installed:

```bash
# Install http-server globally (one time only)
npm install -g http-server

# Run the server
http-server -p 8000
```

Then open your browser to: `http://localhost:8000`

### Option 3: VS Code Live Server

If you use VS Code:
1. Install the "Live Server" extension
2. Right-click on `index.html`
3. Select "Open with Live Server"

## Usage

1. Fill out the daily PANAS questionnaire (rate 20 emotions on a 1-5 scale)
2. Indicate if you're on your period
3. Add any notes about the day
4. Click "Save Entry"
5. View your mood trends over time in the chart
6. Click on any data point to see notes for that day

## Data Storage

- All data is stored in your browser's localStorage
- Your historical CSV data is automatically imported on first load
- Use the "Re-import CSV" button if you need to reload the CSV data

## CSV Format

The app imports data from `PANAS tracking - Sheet1.csv` with columns:
- date (M/D/YY or M/D/YYYY format)
- positive affect (10-50)
- negative affect (10-50)
- sleep hrs (ignored)
- sleep score (ignored)
- period (y/n)
- notes

## Troubleshooting

**"Failed to fetch" error**: You must run the app through a local web server (see "How to Run" above). Opening the HTML file directly won't work due to browser security restrictions.

**Chart not showing data**:
1. Open browser console (F12)
2. Look for import messages
3. Try clicking "Re-import CSV" button
4. Check that CSV file is in the same directory as index.html
