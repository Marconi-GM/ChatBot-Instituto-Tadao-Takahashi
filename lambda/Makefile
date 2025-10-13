# ITT ChatBot Lambda Makefile

.PHONY: build build-no-cache build-multistage clean help

IMAGE_NAME := itt-chatbot
DOCKERFILE := Dockerfile

# Default target
help:
	@echo "ITT ChatBot Lambda Build Commands:"
	@echo "=================================="
	@echo "make build           - Build with cache optimization"
	@echo "make build-no-cache  - Force rebuild without cache"
	@echo "make build-multistage- Use multi-stage Dockerfile"
	@echo "make clean          - Remove built images"
	@echo "make size           - Show image size"
	@echo "make run            - Run container locally for testing"
	@echo ""

# Build with cache (fastest)
build:
	@echo "Building with cache optimization..."
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME) .

# Build without cache (slowest but cleanest)
build-no-cache:
	@echo "Building without cache..."
	docker build --no-cache -t $(IMAGE_NAME) .

# Build with multi-stage Dockerfile (best for production)
build-multistage:
	@echo "Building with multi-stage Dockerfile..."
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.multistage -t $(IMAGE_NAME) .

# Clean up images
clean:
	@echo "Removing built images..."
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
	docker image prune -f

# Show image size
size:
	@docker images $(IMAGE_NAME) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Run container locally for testing
run:
	@echo "Running container locally..."
	docker run --rm -p 9000:8080 $(IMAGE_NAME)

# Build and show stats
build-stats: build size
	@echo "Build completed!"