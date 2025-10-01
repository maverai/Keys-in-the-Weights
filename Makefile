.PHONY: install run reproduce plots

install:
	pip install -r requirements.txt

run:
	python -m src.evaluate

reproduce:
	bash scripts/reproduce.sh

plots:
	python scripts/make_plots.py
.PHONY: clean
clean:
	rm -rf results/*
	rm -rf figures/*