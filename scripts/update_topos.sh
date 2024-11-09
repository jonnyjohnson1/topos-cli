#!/bin/bash

# Shutdown the current topos process
pkill -f topos  # Replace this with a specific shutdown command if needed

# Wait a moment for the shutdown to complete
sleep 2

# Get the latest release (you could fetch this in Python and pass it here)
LATEST_RELEASE=$(poetry run python -c "from topos.utils.check_for_update import get_latest_release_tag; print(get_latest_release_tag('jonnyjohnson1', 'topos-cli'))")

# Start the latest release
nix run github:repo_owner/repo_name/$LATEST_RELEASE