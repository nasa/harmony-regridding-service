###############################################################################
#
# Test image for the Harmony Regridding service. This test image uses the main
# service image, sds/harmony-regridder, as a base layer for the tests. This
# ensures that the contents of the service image are tested, preventing
# discrepancies between the service and test environments.
#
# Updated: 2023-01-26
#
###############################################################################
FROM sds/harmony-regridder

ENV PYTHONDONTWRITEBYTECODE=1

# Install additional Pip requirements (for testing)
COPY tests/pip_test_requirements.txt .
RUN pip install --no-input -r pip_test_requirements.txt

# Copy test directory containing Python unittest suite, test data and utilities 
COPY ./tests tests

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run_tests.sh"]
