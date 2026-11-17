# Makefile for building and deploying JupyterBook

.PHONY: help appendix build build-local clean deploy serve

help:
	@echo "Available commands:"
	@echo "  make appendix     - Execute the voting-rights appendix into jb/ (run before build)"
	@echo "  make build        - Build the JupyterBook (GitHub Pages base URL)"
	@echo "  make build-local  - Build without a base URL (local serve)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make deploy       - Build and deploy to GitHub Pages"
	@echo "  make serve        - Build locally and serve at http://localhost:8000"

# postbuild.py fixes the site address (the theme writes localhost), adds canonical
# links, rewrites the sitemap, and adds the page-view counter (greenteapress.com/gh.gif,
# counted from the greenteapress.com access logs; no cookies, no third party).
# The voting-rights appendix is published as a page with its outputs, so it's executed
# here (from notebooks/, where it finds ../data/voting_rights) into jb/; the copy is not committed.
appendix:
	cd ../notebooks && jupyter nbconvert --to notebook --execute voting_rights_appendix.ipynb \
		--ExecutePreprocessor.timeout=1200 --output-dir ../jb --output voting_rights_appendix

build: appendix
	rm -rf _build/html
	BASE_URL=/YouWouldChooseNow jupyter book build --html
	touch _build/html/.nojekyll
	python postbuild.py _build/html

build-local: appendix
	jupyter book build --html

clean:
	rm -rf _build

deploy: build
	ghp-import -n -p -f _build/html

serve: build-local
	cd _build/html && python -m http.server 8000
