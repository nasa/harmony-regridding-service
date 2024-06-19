# Harmony Regridding Service

This repository contains the source code, unit test suite, Jupyter notebook
documentation and build-related scripts for the Harmony Regridding Service. The
initial version of this service will perform regridding only, for collections
with one geographic grid to another geographic grid, in support of the GES DISC
GPM/IMERGHH and MERRA collections.

## Repository structure:

```
|- CHANGELOG.md
|- CONTRIBUTING.md
|- README.md
|- legacy-CHANGELOG.md
|- bin
|- docker
|- harmony_regridding_service
|- pip_requirements.txt
|- tests
```

* .snyk - A file used by the Snyk webhook to ensure the correct version of
  Python is used when installing the full dependency tree for the Harmony
  Regridding Service. This file should be updated when the version of Python is
  updated in the service Docker image.
* CHANGELOG.md - This file contains a record of changes applied to each new
  public release of a service Docker image. Any release of a new service
  version since migrating to GitHub should have a record of what was changed in
  this file (e.g., starting at version 1.0.0).
* CONTRIBUTING.md - General guidelines for making contributions to the
  repository.
* LICENSE - The NASA open-source license under which this software has been
  made available.
* README.md - This file, containing guidance on developing the service.
* legacy-CHANGELOG.md - This file contains release notes for all versions of
  the service that were released internally to EOSDIS, before migrating the
  service code and Docker image to GitHub.
* bin - A directory containing utility scripts to build the service and test
  images.
* docker - A directory containing the Dockerfiles for the service and test
  images. It also contains `service_version.txt`, which contains the semantic
  version number of the service image. Any time an update is made that should
  have an accompanying service image release, this file should be updated.
* docs - A directory containing Jupyter notebook documentation showing an
  end-user how to use the service.
* harmony_regridding_service - The directory containing Python source code for
  the Harmony Regridding Service. `adapter.py` contains the `RegriddingServiceAdapter`
  class that is invoked by calls to the service.
* pip_requirements.txt - A list of service Python package dependencies.
* tests - A directory containing the service unit test suite.

## Contributing:

Contributions are welcome! For more information, see `CONNTRIBUTING.md`.

## Python dependencies:

At this time, the service only requires dependencies that can be obtained via
Pip. The service dependencies are contained in `pip_requirements.txt`.
Additional test dependencies are listed in `tests/pip_test_requirements.txt`.

## Local development:

Local testing of service functionality is best achieved via a local instance of
[Harmony](https://github.com/nasa/harmony). Please see instructions there
regarding creation of a local Harmony instance.

If testing small functions locally that do not require inputs from the main
Harmony application, it is recommended that you create a Python virtual
environment via tools such as pyenv or conda, and install the necessary
dependencies for the service within that environment via Pip.

## Tests:

This service utilises the Python `unittest` package to perform unit tests on
classes and functions in the service. After local development is complete, and
test have been updated, they can be run via:

```bash
$ ./bin/build-image
$ ./bin/build-test
$ ./bin/run-test
```

The `tests/run_tests.sh` script will also generate a coverage report, rendered
in HTML, and scan the code with `pylint`.

The `unittest` suite is run automatically via GitHub Actions as part of a
GitHub "workflow". These workflows are defined in the `.github/workflows`
directory.

## pre-commit hooks:

This repository uses [pre-commit](https://pre-commit.com/) to enable pre-commit
checking the repository for some coding standard best practices. These include:

* Removing trailing whitespaces.
* Removing blank lines at the end of a file.
* JSON files have valid formats.
* [ruff](https://github.com/astral-sh/ruff) Python linting checks.
* [black](https://black.readthedocs.io/en/stable/index.html) Python code
  formatting checks.

To enable these checks:

```bash
# Install pre-commit Python package as part of test requirements:
pip install -r tests/pip_test_requirements.txt

# Install the git hook scripts:
pre-commit install

# (Optional) Run against all files:
pre-commit run --all-files
```

When you try to make a new commit locally, `pre-commit` will automatically run.
If any of the hooks detect non-compliance (e.g., trailing whitespace), that
hook will state it failed, and also try to fix the issue. You will need to
review and `git add` the changes before you can make a commit.

It is planned to implement additional hooks, possibly including tools such as
`mypy`.

[pre-commit.ci](pre-commit.ci) is configured such that these same hooks will be
automatically run for every pull request.

## Versioning:

Service Docker images for the Harmony Regridding Service adhere to semantic
version numbers: `major.minor.patch`.

* Major increments: These are non-backwards compatible API changes.
* Minor increments: These are backwards compatible API changes.
* Patch increments: These updates do not affect the API to the service.

When publishing a new Docker image for the service, two files need to be
updated:

* CHANGELOG.md - Notes should be added to capture the changes to the service.
* docker/service_version.txt - The semantic version number should be updated.

## Docker image publication:

Service Docker images are published to ghcr.io any time a change is merged to
the `main` branch that contains an update to `docker/service_version.txt`.

## Releasing a new version of the service:

Once a new Docker image has been published with a new semantic version tag,
that service version can be released to a Harmony environment by updating the
main Harmony Bamboo deployment project. Find the environment you wish to
release the service version to and update the associated environment variable
to update the semantic version tag at the end of the full Docker image name.

## Get in touch:

You can reach out to the maintainers of this repository via email:

* david.p.auty@nasa.gov
* owen.m.littlejohns@nasa.gov
* matthew.savoie@colorado.edu
