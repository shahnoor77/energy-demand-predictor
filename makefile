# backfills the feature store with historical data
backfill:
	poetry run python src/component/backfill_feature_group.py
# generates new batch of features and stores them in the feature store
features:
	poetry run python src/pipelines/feature_pipeline.py