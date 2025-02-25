# Enviroment setup

## uv

We use [uv](https://github.com/astral-sh/uv) for packaging. To install it, run:

```bash
pip install uv
```

To install the dependencies, run:

```bash
uv sync
```

See the [uv documentation](https://docs.astral.sh/uv/guides/projects/#running-commands) for a quick start guide on how to use it.

## Ruff

We use [Ruff](https://docs.astral.sh/ruff/) as a linter and code formatter.

It's installed as a dependency, so you don't need to install it manually.

If you're using VS Code, you can install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to check and format your code. It's already configured to use the project's configuration file and to run the formatter on save.
