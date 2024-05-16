#cli.py

import argparse

def main():
    parser = argparse.ArgumentParser(description="CLI for my script")
    parser.add_argument('command', choices=['run', 'dl'], help="Command to execute")
    parser.add_argument('--web', action='store_true', help="Flag to run the server for web access")
    parser.add_argument('--local', action='store_true', help="Flag to run the server for local access (default)")
    parser.add_argument('--spacy', choices=['small', 'med', 'large', 'trf'], help="Specify Spacy model size (only for 'download' command)")

    args = parser.parse_args()

    if args.command == 'run':
        """
        start the topos api server
        """
        # imoprt api
        from .api import api
        if args.web:
            api.start_web_api()
        else:
            api.start_local_api()
    
    elif args.command == 'dl':
        """
        download Spacy model
        """
        # imoprt downloaders
        from .downloaders.spacy_loader import download_spacy_model
        if args.spacy:
            download_spacy_model(args.spacy)
        else:
            print("Please specify Spacy model size using --spacy option.")

if __name__ == "__main__":
    main()