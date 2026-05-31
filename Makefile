# =============================================================================
# AI City View — convenience entry points
# =============================================================================

COMPOSE := docker compose

.DEFAULT_GOAL := help

help: ## Show this help.
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*?##/{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

up: ## Build (if needed) + start the Vision API container.
	$(COMPOSE) up -d
	@echo ""
	@echo "  Vision API:  http://localhost:8000"
	@echo "  API docs:    http://localhost:8000/docs"
	@echo "  Health:      curl http://localhost:8000/health"
	@echo ""
	@echo "Tail startup logs: docker compose logs vision-api -f"

down: ## Stop and remove containers.
	$(COMPOSE) down

logs: ## Tail vision-api logs (Ctrl+C to detach).
	$(COMPOSE) logs vision-api -f

ps: ## Show container status.
	$(COMPOSE) ps

restart: ## Restart vision-api (e.g. after VISION_DEPTH_MODEL change).
	$(COMPOSE) restart vision-api

shell: ## Open a shell inside the vision-api container.
	$(COMPOSE) exec vision-api bash

build: ## Force a fresh build (no cache).
	$(COMPOSE) build --no-cache

.PHONY: help up down logs ps restart shell build
