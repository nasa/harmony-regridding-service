###############################################################################
#
# Service image for ghcr.io/nasa/harmony-regridder, a Harmony backend service
# that transforms Level 3 or Level 4 data to another grid.
#
# This image installs dependencies via Pip. The service code is then copied
# into the Docker image.
#
# 2023-01-26: Added to repository.
#
###############################################################################
FROM python:3.11.9-slim-bullseye

WORKDIR "/home"

# Install static things necessary for building dependencies.
RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y build-essential git

# Install Python dependencies.
COPY ./pip_requirements.txt pip_requirements.txt
RUN pip3 install --no-input -r pip_requirements.txt

# Copy service code.
COPY ./harmony_regridding_service harmony_regridding_service

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "harmony_regridding_service"]
