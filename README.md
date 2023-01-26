[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build](https://github.com/JanKapala/advent_of_code/actions/workflows/python-app.yml/badge.svg)](https://github.com/JanKapala/advent_of_code/actions/workflows/python-app.yml)

# Awakeit

### Mission
Make NPCs in games more alive, interesting and spectacular by awakening their intelligence!

### Conceptual approach
Conceptual framework used for achieving above goal is `Reinforcement Learning`.

### Project setup

sudo apt install -y python3.11-dev
sudo apt install -y python3.11-tk
sudo apt install -y liblzma-dev


- Install [pyenv](https://github.com/pyenv/pyenv)
- sudo apt install -y python3.11-dev python3.11-tk liblzma-dev libsqlite3-dev (necessarily before python 3.11.1 installation(next step))
- Install Python 3.11.1 (via pyenv)
- Install Package manager: [PDM](https://pdm.fming.dev/)

- Install project dependencies: `pdm install`
  - [pre-commit](https://pre-commit.com/) hook is installed automatically after project dependencies.

- Install [PyCharm](https://www.jetbrains.com/pycharm/) and integrate it with the PDM as described [here](https://pdm.fming.dev/latest/usage/pep582/)
  - This project uses Google style docstrings, set it in the Pycharm settings | Tools | Python Integrated Tools and also check following checkboxes:
    - Analyze Python code in docstrings
    - Render external documentation for stdlib

- Install [Graphviz]() with:
  - `sudo apt install graphviz`