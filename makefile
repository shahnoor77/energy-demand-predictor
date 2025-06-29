.PHONE: init features training inference frontend monitoring

# downloads Poetry and installs all dependencies from pyproject.toml
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install


# backfills the feature store with historical data
backfill:
	poetry run python src/component/backfill_feature_group.py
# generates new batch of features and stores them in the feature store
features:
	poetry run python src/pipelines/feature_pipeline.py

# trains a new model and stores it in the model registry
train:
	poetry run python src/pipelines/training_pipeline.py

# generates predictions and stores them in the feature store
inference:
	poetry run python src/pipelines/inference_pipeline.py

# starts the Streamlit app
frontend-app:
	poetry run streamlit run src/frontend.py

monitoring-app:
	poetry run streamlit run src/monitoring_frontend.py

lint:
	@echo "Fixing linting issues..."
	poetry run ruff check --fix .

format:
	echo "Formatting Python code..."
	poetry run ruff format .