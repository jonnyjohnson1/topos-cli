#cli.py

import argparse
from .api import api

def main():
    parser = argparse.ArgumentParser(description="CLI for my script")
    parser.add_argument('command', choices=['run'], help="Command to execute")
    parser.add_argument('--web', action='store_true', help="Flag to run the server for web access")
    parser.add_argument('--local', action='store_true', help="Flag to run the server for local access (default)")


    args = parser.parse_args()

    if args.command == 'run':
        """
        start the topos api server
        """
        if args.web:
            api.start_web_api()
        else:
            api.start_local_api()

if __name__ == "__main__":
    main()