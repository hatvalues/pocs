test:
	poetry run pytest --cov --cov-config=.coveragerc --cov-report term --cov-report html

test_app:
	poetry run pytest tests/app_tests --cov app --cov-config=.coveragerc --cov-report term --cov-report html



update_test:
	UPDATE_FILE_FIXTURES=TRUE poetry run pytest --cov --cov-config=.coveragerc --cov-report term --cov-report html

install:
	poetry install

lint:
	poetry run pylint --disable=R,C app

check:
	poetry run mypy .

format:
	poetry run ruff format .

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ | xargs rm -rf

all: clean install lint format test

python:
	poetry run python

main:
	poetry run python main.py

poetry:
	pipx install poetry

chromium:
	sudo apt-get update && sudo apt-get install -y chromium-browser libglib2.0-0 libnss3 libgconf-2-4 libatk1.0-0 libgtk-3-0 libx11-6 libasound2 libdbus-glib-1-2