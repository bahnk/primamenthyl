PACKAGE=bam_processing

.PHONY: format lint test

format:
	uv run black $(PACKAGE)

lint:
	uv run pylint --disable=too-few-public-methods $(PACKAGE)

test:
	uv run pytest $(PACKAGE)

clean:
	find $(PACKAGE) -name "__pycache__" -type d -exec rm -rfv {} \; || true
	rm -rfv dist/
	rm -rfv $(PACKAGE).egg-info/
	rm -rfv docs/build/
	rm -f docs/make.bat
	rm -rfv .pytest_cache
	rm -rfv build/
