* The `probound` module is contained in [src/probound](src/probound).
* Package configuration is found in
    [pyproject.toml](pyproject.toml) and [setup.cfg](setup.cfg)
* You can use `tox`(https://tox.wiki) to automate linting, testing, and docs

# Style
* Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
* Use {`expt`, `rnd`, `agg`, `ctrb`, `bmd`, `conv`}`_idx` in internal code,
  and {`experiment`, `round`, `aggregate`, `contribution`, `binding_mode`, `conv_layer`}`_index`
  for user-facing code

# Linters and formatters
* [mypy](https://mypy.readthedocs.io)
* [Pylint](https://pylint.readthedocs.io)
* [Black](https://black.readthedocs.io)
* [isort](https://pycqa.github.io/isort)
* Per-linter settings are specified in [pyproject.toml](pyproject.toml)

# Unit testing
* Run all unit tests in [test](test)
    by running `python -m unittest` from the root directory

# Docs
* Docs are built in [Sphinx](https://www.sphinx-doc.org)
    and can be edited in [docs/source](docs/source)
* The [docs/source/_notebooks](docs/source/_notebooks)
    is tracked using [Git LFS](https://github.com/git-lfs/git-lfs)