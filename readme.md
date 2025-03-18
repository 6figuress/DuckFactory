[![codecov](https://codecov.io/gh/6figuress/DuckFactory/graph/badge.svg?token=23COG54BQU)](https://codecov.io/gh/6figuress/DuckFactory)
![GitHub branch status](https://img.shields.io/github/checks-status/6figuress/DuckFactory/main)

# Enviroment setup

## uv

We use [uv](https://github.com/astral-sh/uv) for packaging. To install it, run:

```bash
pip install uv
```

Create a virtual environment and activate it:

```bash
uv venv
source .venv/bin/activate # Linux
.venv\Scripts\activate # Windows
```

To install the dependencies, run:

```bash
uv sync
```
Install the project in editable mode (only needs to be done once):

```bash
pip install -e .
```

The editable install allows to import the project as a package in the Python interpreter, while still being able
to make changes to the code and see the changes reflected without having to reinstall the package. 
It also ensures a consistent environment across different parts of the project.

See the [uv documentation](https://docs.astral.sh/uv/guides/projects/#running-commands) for a quick start guide on how to use it.

## Ruff

We use [Ruff](https://docs.astral.sh/ruff/) as a linter and code formatter.

It's installed as a dependency, so you don't need to install it manually.

If you're using VS Code, you can install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to check and format your code. It's already configured to use the project's configuration file and to run the formatter on save.


# Tests
Before running tests, make sure that the virtual environment is activated and that the project is installed in editable mode (see above).

To run tests in the terminal, run from the project's root directory:

```bash
uv run pytest
```

You can also run tests from VS Code in the "Testing" tab.
