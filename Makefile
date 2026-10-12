up:
	docker compose up -d --build
	@echo "Waiting for app..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -sf http://localhost:8000/health >/dev/null && break; \
		sleep 1; \
	done
	@curl -sf http://localhost:8000/health >/dev/null \
		&& echo "Stack is UP: app :8000, image-analyzer :8001" \
		|| (echo "App did not become healthy in time"; exit 1)

status:
	docker compose ps

logs:
	docker compose logs -f app image-analyzer

down:
	docker compose down --remove-orphans
