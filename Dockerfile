FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy the package files
COPY . /app/

# Install the package
RUN pip3 install -e .

# Create a non-root user with sudo access
RUN apt-get update && \
    apt-get install -y sudo && \
    useradd -m -s /bin/bash agent && \
    echo "agent ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

# Set up the workspace directory
RUN mkdir -p /app/workspace && \
    chown -R agent:agent /app

USER agent

# Add some helpful aliases and environment settings
RUN echo 'export PS1="\[\033[1;36m\]deepdroid\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]$ "' >> ~/.bashrc && \
    echo 'alias ll="ls -la"' >> ~/.bashrc

# Default to bash instead of python entrypoint for interactive use
CMD ["/bin/bash"] 