# Change Log

The Harmony Regridding Service follows semantic versioning. All notable changes
to this project will be documented in this file. The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).


## [v1.1.0] - 2025-05-05

## Added

- Adds Command line entrypoint for testing the service without Docker.
- Adds support and configuraton for input files that have projected
  coordinates. These include the SMAP L4 Collections SPL4CMDL, SPL4SMAU,
  SPL4SMGP and SPL4SMLM.

## Changed

- Internal changes.
  + Python service lib updated to 3.12.
  + Python lib dependencies are updated.
  + Refactors applied.
  + `test_regridding_service.py` converted to pytest.
  + Adds python fixture for SPL4SMAU-like test file.


## [v1.0.6] - 2025-03-24

## Changed

- This version should show no user facing changes. Python libraries are updated
  and inconsequential internal refactorings have been made.


## [v1.0.5] - 2025-03-20

## Changed

- This version should show no user facing changes. The pre-commit and
  linting is updated to a more modern mechanism.

## [v1.0.4] - 2024-09-30

## Changed
This version of the Harmony Regridding Service updates most dependency versions,
most notably updating to `harmony-service-lib-py==2.0.0`. This is a breaking
change in the package, requiring the renaming of imports in the Harmony
Regridding Service.

## [v1.0.3] - 2024-09-26

## Changed
This version of the Harmony Regridding Service updates the version of the
`harmony-service-lib` used to 1.1.0. There are no functional updates in this
version of the Harmony Regridding Service.

## [v1.0.2] - 2024-09-23

### Changed
This version of the Harmony Regridding Service updates to use
`earthdata-varinfo==3.0.0`, which primarily requires updates to the
configuration file the Harmony Regridding Service uses to specify metadata
overrides and excluded science variables to `earthdata-varinfo`.

This update also ensures that the Docker images used for local testing are
named `ghcr.io/nasa/harmony-regridding-service` and
`ghcr.io/nasa/harmony-regridding-service-test`, to comply with the released
image name for the service, and the name that Harmony-in-a-Box is looking for.

## [v1.0.1] - 2024-06-20

### Changed
This version of the Harmony Regridding Service updates to use Python 3.11.

## [v1.0.0] -  2024-04-12

This version of the Harmony Regridding Service contains all functionality
previously released internally to EOSDIS as `sds/harmony-regridder:0.0.4`.
Minor reformatting of the repository structure has occurred to comply with
recommended best practices for a Harmony backend service repository, but the
service itself is functionally unchanged. Additional contents to the repository
include updated documentation and files outlined by the
[NASA open-source guidelines](https://code.nasa.gov/#/guide).

For more information on internal releases prior to NASA open-source approval,
see legacy-CHANGELOG.md.

[v1.0.6]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.6
[v1.0.5]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.5
[v1.0.4]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.4
[v1.0.3]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.3
[v1.0.2]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.2
[v1.0.1]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.1
[v1.0.0]: https://github.com/nasa/harmony-regridding-service/releases/tag/1.0.0
