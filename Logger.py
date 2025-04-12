import os
import logging
import logging.handlers
import sys
from Singleton import Singleton

class Logger(metaclass=Singleton):

    os.makedirs('logs', exist_ok=True)

    def __init__(self, logger_name, level: str = "ERROR"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

        file_handler = logging.handlers.RotatingFileHandler(
            filename=f'logs/{logger_name}.log',
            backupCount=2,
            maxBytes=10240 # 10KB
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)
