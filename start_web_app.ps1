# Start the Fruit Recognition Web Application for Windows PowerShell

Write-Host "🚀 Starting Fruit Recognition Web App..." -ForegroundColor Green
Write-Host ""
Write-Host "📱 Open your browser and go to:" -ForegroundColor Cyan
Write-Host "   http://localhost:5000" -ForegroundColor Cyan
Write-Host "   OR" -ForegroundColor Cyan
Write-Host "   http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "📸 Upload a fruit image to get predictions!"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

python app.py
