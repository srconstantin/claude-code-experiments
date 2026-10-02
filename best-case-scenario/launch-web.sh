#!/bin/bash
# Starts the browser version of Best-Case Scenario in a new macOS Terminal window
# (the window runs the local server; the app itself opens in your browser).
# Used by "/best-case web" in Claude Code; also fine to run by hand.
DIR="$(cd "$(dirname "$0")" && pwd)"
osascript <<OSA
tell application "Terminal"
    activate
    do script "cd '$DIR' && clear && python3 web.py"
end tell
OSA
