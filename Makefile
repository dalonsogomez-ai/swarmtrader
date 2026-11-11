.PHONY: install test rank validate tournament publish swarm nulltest infogeo

PY ?= python3

install:            ## crea .venv e instala (macOS/Linux)
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[data,dev,report]"

test:               ## tests
	. .venv/bin/activate && pytest -q

rank:               ## make rank CSV=mi_universo.csv TOP=5
	. .venv/bin/activate && swarmtrader rank $(or $(CSV),examples/universe_tickers.csv) --top $(or $(TOP),5)

validate:           ## controles de overfitting (obligatorio antes de creerse nada)
	. .venv/bin/activate && swarmtrader validate

tournament:         ## torneo evolutivo completo con la config por defecto
	. .venv/bin/activate && swarmtrader tournament --config config/default.toml

publish:            ## crea el repo en GitHub y hace push (requiere gh)
	bash scripts/publish.sh

swarm:            ## enjambre sobre CSV=...
	$(PY) -m swarmtrader.cli swarm $(CSV) --top 5

nulltest:         ## los dos tests de control (ruido y potencia)
	$(PY) scripts/null_test.py
	$(PY) scripts/swarm_null_test.py

infogeo:          ## geometria de la informacion: tamano/potencia del monitor y torneo de distancias
	$(PY) experiments/infogeo.py $(if $(PANEL),--stocks-panel $(PANEL),)
