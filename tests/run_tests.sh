#!/bin/sh
###############################################################################
#
# A script invoked by the test Dockerfile to run the Python test suite
# for the Harmony Regridding Service. The script first runs the test suite,
# then it checks for linting errors.
###############################################################################

# Exit status used to report back to caller
STATUS=0

# Run the standard set of unit tests, producing JUnit compatible output
pytest --cov=harmony_regridding_service \
       --cov-report=html:reports/coverage \
       --cov-report term \
       --junitxml=reports/test-reports/test-results-"$(date +'%Y%m%d%H%M%S')".xml || STATUS=1


# Run pylint
pylint harmony_regridding_service
RESULT=$((3 & $?))

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: pylint generated errors"
fi

exit $STATUS
