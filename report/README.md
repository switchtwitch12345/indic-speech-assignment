Build the PDF (needs Google Chrome):

    python report/build_report.py
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
        --no-pdf-header-footer --print-to-pdf=report/report.pdf report/report.html
