.PHONY: help install test format lint clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run unit tests"
	@echo "  make format   - Format code (placeholder)"
	@echo "  make lint     - Lint code (placeholder)"
	@echo "  make clean    - Clean generated files"
	@echo "  make example  - Run example script"
	@echo "  make run      - Run CSV question processor"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

format:
	@echo "Format not configured - install ruff if needed"
	# ruff format .

lint:
	@echo "Lint not configured - install ruff if needed"
	# ruff check .

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	rm -rf test_documents/
	rm -rf vector_store/
	rm -rf document_store/
	rm -f answers.txt

example:
	python example.py

run:
	@echo "Usage: python run_qa_pipeline.py [questions.csv]"
	@echo "Update document_storage_dir in run_qa_pipeline.py first!"
