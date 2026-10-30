# Makefile for building and deploying JupyterBook

.PHONY: help build build-local clean deploy serve

help:
	@echo "Available commands:"
	@echo "  make build        - Build the JupyterBook (GitHub Pages base URL)"
	@echo "  make build-local  - Build without a base URL (local serve)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make deploy       - Build and deploy to GitHub Pages"
	@echo "  make serve        - Build locally and serve at http://localhost:8000"

# postbuild.py fixes the site address (the theme writes localhost), adds canonical
# links, rewrites the sitemap, and adds the page-view counter (greenteapress.com/gh.gif,
# counted from the greenteapress.com access logs; no cookies, no third party).
build:
	rm -rf _build/html
	BASE_URL=/YouWouldChooseNow/ jupyter book build --html
	touch _build/html/.nojekyll
	python postbuild.py _build/html

build-local:
	jupyter book build --html

clean:
	rm -rf _build

deploy: build
	ghp-import -n -p -f _build/html

serve: build-local
	cd _build/html && python -m http.server 8000
