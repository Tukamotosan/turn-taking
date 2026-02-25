#!/bin/bash

# Get the project root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$DIR/.." && pwd )"

# Initial docker run command
DOCKER_CMD="docker run --rm -it --gpus all \
    --user $(id -u):$(id -g) \
    --volume $PROJECT_ROOT:/home/ubuntu/turn-taking"

# Process additional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --volume)
            if [ -n "$2" ]; then
                # Split from:to
                VOLUME_FROM="${2%%:*}"
                VOLUME_TO="${2#*:}"
                
                if [ -d "$VOLUME_FROM" ]; then
                    DOCKER_CMD="$DOCKER_CMD --volume $VOLUME_FROM:$VOLUME_TO"
                else
                    echo "Warning: Directory $VOLUME_FROM does not exist. Skipping mount to $VOLUME_TO."
                fi
                shift
            fi
            ;;
        *)
            # Other arguments are passed as the command to run in the container
            EXTRA_ARGS+="$1 "
            ;;
    esac
    shift
done

# Execute the final command
# Use bash as the default entrypoint if no extra arguments are provided
if [ -z "$EXTRA_ARGS" ]; then
    EXTRA_ARGS="bash"
fi

$DOCKER_CMD turn-taking:latest $EXTRA_ARGS
