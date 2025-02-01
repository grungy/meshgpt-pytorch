# Makefile for managing Docker Compose

COMPOSE_FILE=docker-compose.yml

# Default target (optional)
.PHONY: up
up:
	docker compose -f $(COMPOSE_FILE) up -d --build

.PHONY: down
down:
	docker compose -f $(COMPOSE_FILE) down

.PHONY: cleanImages
cleanImages:
	# Stops containers and removes all images created by the Compose file
	docker compose -f $(COMPOSE_FILE) down --rmi all

.PHONY: prune
prune:
	# Removes all unused Docker images, containers, and networks
	docker system prune -af

.PHONY: rebuild
rebuild:
	# Rebuilds all images and starts containers
	docker compose -f $(COMPOSE_FILE) up --build -d

.PHONY: worker
worker:
	# starts the InFactory Worker
	poetry run python infactory_api/worker.py

.PHONY: backend
backend:
	# start the InFactory backend
	./start.sh

.PHONY: cleanEnv
cleanEnv:
	# Clean the environment
	lsof -ti :8000 | xargs kill -9 & deactivate & python3 -m venv venv & source venv/bin/activate & pip install . & rm -rf node_modules/.prisma