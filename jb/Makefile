# Makefile for building and deploying JupyterBook

.PHONY: help build build-local clean deploy serve

help:
	@echo "Available commands:"
	@echo "  make build        - Build the JupyterBook (GitHub Pages base URL)"
	@echo "  make build-local  - Build without a base URL (local serve)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make deploy       - Build and deploy to GitHub Pages"
	@echo "  make serve        - Build locally and serve at http://localhost:8000"

build:
	BASE_URL=/YouWouldChooseNow/ jupyter book build --html
	touch _build/html/.nojekyll

build-local:
	jupyter book build --html

clean:
	rm -rf _build

deploy: build
	ghp-import -n -p -f _build/html

serve: build-local
	cd _build/html && python -m http.server 8000
