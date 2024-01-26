export_requirements:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

install_requirements:
	pip install -r requirements.txt

run_classical:
	python3 classical.py
