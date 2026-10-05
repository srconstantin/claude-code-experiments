#!/bin/bash
# Starts the Grammar of Ornament server in a new macOS Terminal window
# (the window runs the local server; the app itself opens in your browser).
DIR="$(cd "$(dirname "$0")" && pwd)"
osascript <<OSA
tell application "Terminal"
    activate
    do script "cd '$DIR' && clear && .venv/bin/python server.py"
end tell
OSA
