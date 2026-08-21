#!/usr/bin/env bash
set -e

PORT=8000
echo "================================================================="
echo "   CHIMERA-Agent Clinical Interpretability Dashboard"
echo "================================================================="
echo "Starting ultra-lightweight local dashboard server on port ${PORT}..."
echo "Open your browser to: http://localhost:${PORT}/docs/"
echo "Press Ctrl+C to terminate."
echo "================================================================="

cd "$(dirname "$0")/.."
python3 -m http.server ${PORT}
