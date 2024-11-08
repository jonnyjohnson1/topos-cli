#cli.py

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="CLI for my script")
    parser.add_argument('command', choices=['run', 'set', 'chat', 'zrok'], help="Command to execute")
    parser.add_argument('--web', action='store_true', help="Flag to run the server for web access")
    parser.add_argument('--local', action='store_true', help="Flag to run the server for local access (default)")
    parser.add_argument('--spacy', choices=['small', 'med', 'large', 'trf'], help="Specify Spacy model size (only for 'set' command)")
    parser.add_argument('--cloud', action='store_true', help="Flag to run the server on cloud")

    args = parser.parse_args()
    if args.command == 'run':
        """
        start the topos api server
        """
        # import api
        from .api import api
        if args.cloud:
            api.start_hosted_service()
        elif args.web:
            from .app import menu_bar_app
            menu_bar_app.start_web_app()
        else:
            from .app import menu_bar_app
            menu_bar_app.start_local_app()
    
    if args.command == 'chat':
        """
        start the topos api chat server for clients to connect
        """
        # import chat_api
        from .chat_api import api
        api.start_messenger_server()
    
    if args.command == 'zrok':
        """
        start the topos api chat server for clients to connect
        """
        # start zrok server
        import subprocess
        # zrok share public http://0.0.0.0:13341 # the cli command
        subprocess.run(['zrok', 'share', 'public', 'http://0.0.0.0:13341'], check=True)
        
    
    elif args.command == 'set':
        """
        download Spacy model
        """
        # import downloaders
        from .downloaders.spacy_loader import download_spacy_model
        if args.spacy:
            download_spacy_model(args.spacy)
        else:
            print("Please specify Spacy model size using --spacy option.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
