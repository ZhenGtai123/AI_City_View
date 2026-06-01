# Cross-platform "up" wrapper for Windows users without `make`.
# Brings the Vision API container up detached, then prints the URL.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

docker compose up -d

Write-Host ""
Write-Host "  Vision API:  http://localhost:8000"
Write-Host "  API docs:    http://localhost:8000/docs"
Write-Host "  Health:      curl http://localhost:8000/health"
Write-Host ""
Write-Host "Tail logs:   docker compose logs vision-api -f"
Write-Host "Stop:        docker compose down"
