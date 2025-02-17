#!/bin/bash

CONTAINER_NAME="deepdroid"

# Check if the container already exists
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
        # Cleanup old container
        docker rm $CONTAINER_NAME
    fi
    
    # Run new container
    docker run -it -d \
        --name $CONTAINER_NAME \
        --network host \
        -v $(pwd)/workspace:/app/workspace \
        deepdroid:latest
fi

# If no arguments provided, attach to the container
if [ $# -eq 0 ]; then
    docker exec -it $CONTAINER_NAME /bin/bash
else
    # Otherwise run the provided command
    docker exec -it $CONTAINER_NAME "$@"
fi 