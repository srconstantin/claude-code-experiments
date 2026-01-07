# PANAS Mood Tracker

A web application for tracking daily mood using the PANAS (Positive and Negative Affect Schedule) questionnaire.

## Features

- Complete PANAS questionnaire with 20 emotion items
- Period tracking and daily notes
- Interactive line chart showing positive and negative affect over time
- Period days highlighted with light red shading
- Click on data points to view notes
- CSV import for historical data
- Data stored locally in browser

## How to Use

Simply open `index.html` in your web browser. No server required!

## Usage

### Recording Daily Entries

1. Fill out the daily PANAS questionnaire (rate 20 emotions on a 1-5 scale)
2. Indicate if you're on your period (yes/no)
3. Add any notes about the day
4. Click "Save Entry"

### Viewing Your Mood History

- View your mood trends over time in the interactive chart
- Green line shows positive affect scores (10-50)
- Red line shows negative affect scores (10-50)
- Period days are highlighted with light red shading
- Click on any data point to see notes for that day

### Importing Historical Data

If you have historical PANAS data in CSV format:

1. Click the "Import CSV Data" button
2. Select your CSV file
3. Data will be merged with your existing entries

**CSV Format:**
Your CSV should have these columns (in order):
- date (M/D/YY or M/D/YYYY format, e.g., 8/23/24 or 11/7/2024)
- positive affect (10-50)
- negative affect (10-50)
- sleep hrs (optional, will be ignored)
- sleep score (optional, will be ignored)
- period (y/n)
- notes (optional)

**Example CSV:**
```csv
date,positive affect,negative affect,sleep hrs,sleep score,period,notes
8/23/24,41,11,6.1,76,n,pool trip
8/24/24,37,11,7.5,70,n,
8/25/24,36,21,6.9,75,y,feeling tired
```

## Data Storage

- All data is stored locally in your browser's localStorage
- Your data never leaves your computer
- Clearing browser data will delete your entries

## Privacy

This app runs entirely in your browser. No data is sent to any server or third party. All your mood tracking data stays private on your device.
