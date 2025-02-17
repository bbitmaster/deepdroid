FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for Python and PATH
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONPATH="/app"
ENV DEEPDROID_ROOT="/app/deepdroid"

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg \
    gpg-agent \
    && apt-get update \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
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

# Ensure pip is properly set up
RUN python -m ensurepip --upgrade && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ln -sf /usr/local/bin/pip /usr/bin/pip && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip3

# Set up application directories
RUN mkdir -p /app/workspace /app/data

# Install the package
WORKDIR /app
COPY . /app/

# Verify system prompts exist
RUN test -f /app/deepdroid/system_prompts/codebase_improver.txt || (echo "System prompts not found" && exit 1)

# Install the package and ensure it's in PATH
RUN python -m pip install --no-cache-dir -e . && \
    python -c "import pkg_resources; print(pkg_resources.get_distribution('deepdroid').get_entry_info('console_scripts', 'deepdroid').module_name)" > /tmp/entry_point && \
    echo '#!/bin/bash' > /usr/local/bin/deepdroid && \
    echo "PYTHONPATH=/app python -m $(cat /tmp/entry_point) \"\$@\"" >> /usr/local/bin/deepdroid && \
    chmod +x /usr/local/bin/deepdroid && \
    ln -sf /usr/local/bin/deepdroid /usr/bin/deepdroid

# Create a non-root user with sudo access
RUN apt-get update && \
    apt-get install -y sudo && \
    useradd -m -s /bin/bash agent && \
    echo "agent ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

# Set permissions
RUN chown -R agent:agent /app

# Switch to the agent user
USER agent

# Set up shell environment and PATH
RUN echo 'export PS1="\[\033[1;36m\]deepdroid\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]$ "' >> ~/.bashrc && \
    echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PYTHONPATH="/app:$PYTHONPATH"' >> ~/.bashrc && \
    echo 'export DEEPDROID_ROOT="/app/deepdroid"' >> ~/.bashrc

# Set the workspace as the default working directory
WORKDIR /app/workspace

# Default to bash instead of python entrypoint for interactive use
CMD ["/bin/bash"] 