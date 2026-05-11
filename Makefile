.PHONY: clean

clean:
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d \( -name '.tmp' -o -name '.venv' \) -prune -exec rm -rf {} +
	find dbt_models -type d \( -name 'target' -o -name 'logs' \) -prune -exec rm -rf {} +
