import logging
import atexit
from pathlib import Path
from drpd.config import app_config

class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger("drpd-logger")
        self._setup()

    def _setup(self):
        if self.logger.handlers: # đảm bảo mỗi logger chỉ được setup 1 lần duy nhất
            return

        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # đảm bảo dòng log độc lập không liên quan đếm dòng log trong chuỗi log phức tạp

        #console_handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )

        #file handler
        log_file_path = Path(app_config["app"]["log_file"])
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        atexit.register(lambda: [h.close() for h in self.logger.handlers])
    
    def info_console(self, message):
        temp_logger = logging.getLogger("console_only")
        if not temp_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            temp_logger.addHandler(handler)
            temp_logger.setLevel(logging.INFO)
        temp_logger.info(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)

    
custom_logger = CustomLogger()
    
