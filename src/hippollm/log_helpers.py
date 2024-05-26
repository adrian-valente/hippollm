import datetime
import logging
import orjson
import os

def setup() -> None:
    """Set up the logging configuration"""
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(os.getcwd(), f'hippodb_{dt}.log'),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
setup()

def log_action(action: str, prompt: (str|list), answer: (str|list), **kwargs) -> None:
    """Log an action to the console"""
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
    logging.info(f"""MESSAGE:{message}""")
    print(message)