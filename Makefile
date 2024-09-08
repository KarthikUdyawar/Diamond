# Variables
VENV_DIR = ~/Projects/Diamond/venv
DOCKER_COMPOSE_FILE = containers/notebook-docker-compose.yml
PROJECT_NAME = diamond

# Create virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

# Activate virtual environment (this is informational; you can't activate the venv in a Makefile)
.PHONY: activate
activate:
	@echo "Run the following to activate the virtual environment:"
	@echo "source $(VENV_DIR)/bin/activate"

# Run Jupyter notebook
.PHONY: notebook
notebook:
	jupyter notebook

# Build and run Jupyter notebook container
.PHONY: docker-notebook
docker-notebook:
	docker-compose -p $(PROJECT_NAME) -f $(DOCKER_COMPOSE_FILE) up --build -d

# Stop Jupyter notebook container
.PHONY: docker-stop
docker-stop:
	docker-compose -p $(PROJECT_NAME) -f $(DOCKER_COMPOSE_FILE) down

# Clean stop/remove containers
.PHONY: clean
clean:
	# Stop and remove Docker containers
	@echo "Stopping and removing Docker containers for $(PROJECT_NAME)..."
	docker-compose -p $(PROJECT_NAME) -f $(DOCKER_COMPOSE_FILE) down -v --rmi all --remove-orphans
	@echo "Clean-up complete."