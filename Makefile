.PHONY: clean readme-pdf

clean:
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d \( -name '.tmp' -o -name '.venv' \) -prune -exec rm -rf {} +
	find dbt_models -type d \( -name 'target' -o -name 'logs' \) -prune -exec rm -rf {} +

readme-pdf:
	find . \
		\( -type d \( -name '.venv' -o -name '.tmp' -o -name '__pycache__' -o -name '.pytest_cache' -o -name 'target' -o -name 'logs' \) -prune \) -o \
		\( -type f -name 'README.md' -exec sh -c 'for file do pandoc "$$file" -o "$${file%.md}.pdf"; done' sh {} + \)
