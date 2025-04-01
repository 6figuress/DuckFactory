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

# Modules
## dither_class.py
This module provides functionality for applying dithering to images, reducing their color palette while preserving visual detail. 

```python
Dither(factor: float, algorithm: str, nc: int)
```
* factor: Scaling factor for the input image (value between 0 and 1).
* algorithm: Dithering algorithm to use:
  *   ```"fs"``` for Floyd-Steinberg dithering (more advanced).
  *   ```"simplePalette"``` for a basic palette-mapping approach.
```
apply_dithering(image)
```
Applies the selected dithering algorithm to the input image.
* Input: An image (PIL or compatible format).
* Output: A new image with the dithering effect applied.
## path_bounder.py
```python
PathBounder(
  mesh: Trimesh,
  analyzer: PathAnalyzer = None,
  model_points: Positions = None,
  nz_threshold: float = 0.0,
  step_size: float = 0.05,
  precision: float = 1e-6,
  bbox_scale: float = 1.0
)
```
* mesh: A Trimesh object representing the geometry.
* analyzer: Optional PathAnalyzer for surface normal analysis.
* model_points: Optional list of 3D model points used in analysis.
* nz_threshold: Threshold for the Z-component of normals triggering adjustment.
* step_size: Step resolution for path traversal.
* precision: Precision for rounding intersection coordinates.
* bbox_scale: Scale factor applied to the bounding box.
This class enables generation of orientation-aware, constrained paths between 3D points.
```
merge_all_paths(
    paths: list[Path],
    restricted_faces: list[int] = None
) -> list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]
```
Merges multiple paths into one continuous path, optionally excluding specific mesh faces.


