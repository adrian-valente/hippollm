import datetime
import logging
import orjson
import os
from typing import Optional

from .helpers import getroot

_is_setup = False
_path = None

def log_setup(path: Optional[os.PathLike] = None) -> None:
    """Set up the logging configuration"""
    global _is_setup, _path
    if _is_setup and (path is None or path == _path):
        return
    if path is None:
        path = os.path.join(getroot(), 'logs')
    if not os.path.exists(path):
        os.makedirs(path)
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(path, f'hippodb_{dt}.log'),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    _is_setup = True
    

def check_setup() -> None:
    """Check if the logging configuration is set up"""
    if not _is_setup:
        log_setup()    
    

def log_action(action: str, prompt: (str|list), answer: (str|list), **kwargs) -> None:
    """Log an action to the console"""
    check_setup()
    logging.info(f"""ACTION:{
        orjson.dumps({
            'action': action,
            'prompt': str(prompt),
            'answer': str(answer),
            **kwargs
        }).decode()}"""
    )
    
def log_message(message: str) -> None:
    """Log a message to the console and print it to the console."""
    check_setup()
    logging.info(f"""MESSAGE:{message}""")
    print(message)