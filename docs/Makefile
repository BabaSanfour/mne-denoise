# Minimal Sphinx makefile

.PHONY: help clean html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean    remove built documentation"
	@echo "  html     build HTML documentation"

clean:
	@rm -rf _build

html:
	@sphinx-build -M html . _build -W --keep-going
