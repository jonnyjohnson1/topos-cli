import subprocess
import requests
import logging

# Configure logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


repo_owner = "jonnyjohnson1"
repo_name = "topos-cli"

def get_local_version_with_poetry():
    """
    Uses Poetry CLI to get the local version from pyproject.toml.
    """
    result = subprocess.run(["poetry", "version", "--short"], capture_output=True, text=True, check=True)
    local_version = "v" + result.stdout.strip()
    # logger.debug(f"Local version from Poetry: {local_version}")
    return local_version

def get_latest_release_tag(repo_owner, repo_name):
    """
    Fetches the latest release tag from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
    try:
        response = requests.get(url)
        response.raise_for_status()
        latest_release = response.json()["tag_name"]
        # logger.debug(f"Latest release from GitHub: {latest_release}")
        return latest_release
    except requests.ConnectionError:
        logger.debug("No internet connection. Cannot check for the latest release.")
        return None

def check_for_update(repo_owner, repo_name) -> bool:
    """
    Checks if a new release is available by comparing the local version
    obtained from Poetry with the latest GitHub release.
    Returns False if no update is needed or if there is no internet connection.
    """
    try:
        local_version = get_local_version_with_poetry()
        latest_release = get_latest_release_tag(repo_owner, repo_name)
        
        # If we couldn't fetch the latest release (e.g., no internet), assume no update
        if latest_release is None:
            return False

        if local_version == latest_release:
            logger.debug("You have the latest release.")
            return False
        else:
            logger.debug(f"New release available: {latest_release} (current version: {local_version})")
            logger.debug("Consider updating to the latest release.")
            return True
    
    except Exception as e:
        logger.debug(f"An error occurred: {e}")
        return False

def update_topos():
    subprocess.run(["./update_topos.sh"])