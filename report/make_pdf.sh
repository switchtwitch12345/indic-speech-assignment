#!/bin/bash
# Build report.html then print to PDF; stops if the HTML build fails (so a stale PDF is never produced silently).
set -e
python report/build_report.py
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=report/report.pdf report/report.html 2>&1 | tail -1
cp report/report.pdf report/Speech_AI_Assignment_Report.pdf
