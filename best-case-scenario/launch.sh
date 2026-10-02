#!/bin/bash
# Opens the Best-Case Scenario chatbot in a new macOS Terminal window.
# Used by the /best-case slash command in Claude Code; also fine to run by hand.
DIR="$(cd "$(dirname "$0")" && pwd)"
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$DIR' && clear && python3 bcs.py"
end tell
EOF
