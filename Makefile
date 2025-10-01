.PHONY: install run reproduce

install:
	pip install -r requirements.txt

run:
	python -m src.evaluate

reproduce:
	bash scripts/reproduce.sh
# --- IGNORE ---
# End of Makefile