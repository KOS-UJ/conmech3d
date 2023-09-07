#!/bin/sh
# Using .venv

screen -XS examples quit
screen -dmS examples bash -c 'cd ..; source .venv/bin/activate; clear; PYTHONPATH=. python examples/draft_compare.py --shell; exec bash'
