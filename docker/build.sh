#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$DIR/.." && pwd )"

# Build the docker image
# Using the project root as context to allow for COPY/ADD if needed in the future
docker build -t turn-taking:latest "$DIR"
