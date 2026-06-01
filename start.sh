#!/usr/bin/env bash
# Cross-platform "up" wrapper. Brings the Vision API container up
# detached, then prints the URL the user should open.
set -e

cd "$(dirname "$0")"

docker compose up -d

echo ""
echo "  Vision API:  http://localhost:8000"
echo "  API docs:    http://localhost:8000/docs"
echo "  Health:      curl http://localhost:8000/health"
echo ""
echo "Tail logs:   docker compose logs vision-api -f"
echo "Stop:        docker compose down"
