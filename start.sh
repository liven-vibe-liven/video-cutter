#!/bin/bash
set -e
cd "$(dirname "$0")"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ℹ️  ANTHROPIC_API_KEY not set — AI Suggest will be unavailable."
  echo "   To enable: export ANTHROPIC_API_KEY='sk-ant-...'"
else
  echo "✅ ANTHROPIC_API_KEY detected — AI Suggest enabled."
fi

echo "🚀 Starting server at http://localhost:8000"
venv/bin/python3 -m uvicorn app:app --port 8000 --loop asyncio --http h11
