# Makefile for building and deploying JupyterBook

.PHONY: help build build-local clean deploy serve

help:
	@echo "Available commands:"
	@echo "  make build        - Build the JupyterBook (GitHub Pages base URL)"
	@echo "  make build-local  - Build without a base URL (local serve)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make deploy       - Build and deploy to GitHub Pages"
	@echo "  make serve        - Build locally and serve at http://localhost:8000"

# Page-view counter: each page loads greenteapress.com/gh.gif, counted from the
# greenteapress.com access logs (GreenTeaPress scripts/log_report.py). No cookies,
# no third party. Jupyter Book 2 has no setting for this, so it is added after the build.
export COUNTER = <script>new Image().src="https://greenteapress.com/gh.gif?u="+encodeURIComponent(location.href)+"&r="+encodeURIComponent(document.referrer)</script>

build:
	BASE_URL=/YouWouldChooseNow/ jupyter book build --html
	touch _build/html/.nojekyll
	find _build/html -name '*.html' -exec grep -L 'greenteapress.com/gh.gif' {} + | \
		xargs -r perl -pi -e 's#</body>#$$ENV{COUNTER}</body>#'

build-local:
	jupyter book build --html

clean:
	rm -rf _build

deploy: build
	ghp-import -n -p -f _build/html

serve: build-local
	cd _build/html && python -m http.server 8000
