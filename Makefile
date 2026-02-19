PANDOC ?= pandoc
PDF_ENGINE ?= pdflatex
MD_INPUT ?= docs/theorie.md
PDF_OUTPUT ?= docs/theorie.pdf
HTML_OUTPUT ?= docs/theorie.html
MD_FORMAT ?= markdown+tex_math_single_backslash
DOC_TITLE ?= Partie theorique - Optimisation du genou

.PHONY: pdf html docs clean-docs test

pdf:
	$(PANDOC) -f $(MD_FORMAT) $(MD_INPUT) -o $(PDF_OUTPUT) --pdf-engine=$(PDF_ENGINE)

html:
	$(PANDOC) -f $(MD_FORMAT) $(MD_INPUT) -s -o $(HTML_OUTPUT) --mathjax --metadata title="$(DOC_TITLE)"

docs: pdf html

clean-docs:
	rm -f $(PDF_OUTPUT) $(HTML_OUTPUT)

test:
	python3 -m pytest -q
