import secrets
from datetime import datetime


def get_current_time(format="%y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(format)

def get_seed():
    return secrets.randbelow(2147483647)