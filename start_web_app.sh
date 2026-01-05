#!/bin/bash
# Start the Fruit Recognition Web Application

cd "$(dirname "$0")"

# Check if port 5000 is in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 5000 is already in use!"
    echo "💡 Killing existing process..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 2
fi

echo "🚀 Starting Fruit Recognition Web App..."
echo ""
echo "📱 Open your browser and go to:"
echo "   http://localhost:5000"
echo "   OR"
echo "   http://127.0.0.1:5000"
echo ""
echo "📸 Upload a fruit image to get predictions!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py

