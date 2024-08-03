from .api import api

def start_api():
    api.start_hosted_service()

def topos():
    start_api()