# Harmony Regridding Service

This repository contains the source code, unit test suite, Jupyter notebook
documentation and build-related scripts for the Harmony Regridding Service. The
initial version of this service will perform regridding only, for collections
with one geographic grid to another geographic grid, in support of the GES DISC
GPM/IMERGHH and MERRA collections.

## Repository structure:

```
|- CHANGELOG.md
|- README.md
|- bin
|- docker
|- harmony_regridding_service
|- pip_requirements.txt
|- tests
```

* CHANGELOG.md - This file contains a record of changes applied to each new
  release of a service Docker image. Any release of a new service version
  should have a record of what was changed in this file.
* README.md - This file, containing guidance on developing the service.
* bin - A directory containing utility scripts to build the service and test
  images. This also includes scripts that Bamboo uses to deploy new service
  images to AWS ECR.
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

Currently, the `unittest` suite is run automatically within Bamboo as part of a
CI/CD pipeline. In future, this project will be migrated from Bitbucket to
GitHub, at which point the CI/CD will be migrated to workflows that use GitHub
Actions.

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

## Versioning:

Service Docker images for the Harmony Regridding Service adhere to semantic
version numbers: major.minor.patch.

* Major increments: These are non-backwards compatible API changes.
* Minor increments: These are backwards compatible API changes.
* Patch increments: These updates do not affect the API to the service.

When publishing a new Docker image for the service, two files need to be
updated:

* CHANGELOG.md - Notes should be added to capture the changes to the service.
* docker/service_version.txt - The semantic version number should be updated.

## Docker image publication:

Initially service Docker images will be hosted in AWS Elastic Container
Registry (ECR). When this repository is migrated to the NASA GitHub
organisation, service images will be published to ghcr.io, instead.

## Releasing a new version of the service:

Once a new Docker image has been published with a new semantic version tag,
that service version can be released to a Harmony environment by updating the
main Harmony Bamboo deployment project. Find the environment you wish to
release the service version to and update the associated environment variable
to update the semantic version tag at the end of the full Docker image name.
