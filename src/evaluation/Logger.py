import logging
import os
from datetime import datetime
from tqdm import tqdm  # Import tqdm

# --- New: Tqdm-compatible logging handler ---
class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that writes records to the console using tqdm.write(),
    ensuring that log messages do not interfere with tqdm progress bars.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end='\n') # Use tqdm.write to print the message
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class Logger:
    def __init__(self, log_file_path=None, logger_name="LoggerName", level=logging.DEBUG):
        """
        Initializes the logger.

        Args:
            log_file_path (str, optional): Path to the log file. If None, only console logging is enabled.
            logger_name (str, optional): The name of the logger. Defaults to "LoggerName".
            level (int, optional): The minimum logging level for the file handler. Defaults to logging.DEBUG.
        """
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level) # Set the lowest level to capture all messages

        # Avoid adding handlers multiple times if the logger already exists
        if not self._logger.handlers:
            # --- Modified: Use TqdmLoggingHandler for console output ---
            console_handler = TqdmLoggingHandler()
            # Set the level for console output (e.g., INFO to avoid too much noise)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            console_handler.setFormatter(console_formatter)
            self._logger.addHandler(console_handler)

            # --- Unchanged: File handler for detailed logging ---
            if log_file_path:
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
                # The file handler can have a more verbose level
                file_handler.setLevel(level) 
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(file_formatter)
                self._logger.addHandler(file_handler)

    # --- Logger methods (Unchanged) ---
    @property
    def logger(self):
        """Provides direct access to the underlying logger object if needed."""
        return self._logger
        
    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self._logger.exception(message, *args, **kwargs)

    def separator(self, char="=", length=80, message="", level=logging.INFO):
        separator_line = char * length
        if message:
            msg_len = len(message)
            if msg_len < length - 4:
                fill_len = (length - msg_len - 2) // 2
                separator_line = f"{char * fill_len} {message} {char * (length - msg_len - 2 - fill_len)}"
            else:
                separator_line = f"{message[:length]}"
        
        # Log the separator at the specified level
        self._logger.log(level, separator_line)


# --- Translated Test Code ---
if __name__ == "__main__":
    log_directory = "my_app_logs"
    os.makedirs(log_directory, exist_ok=True)
    current_log_file = os.path.join(log_directory, f"application_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Initialize logger with a file path
    my_logger = Logger(logger_name="MyAwesomeApp", log_file_path=current_log_file)

    my_logger.separator(message="Application Start", char="*")
    my_logger.info("This is an important informational log.")
    my_logger.debug("This is a debug message. It won't show on the console by default, but will be in the log file.")
    my_logger.warning("A warning was detected: A parameter might be misconfigured.")

    try:
        result = 10 / 0
    except ZeroDivisionError:
        my_logger.error("Calculation error: division by zero!")
        my_logger.exception("Detailed exception information follows:")

    my_logger.separator(char="-", length=60)
    my_logger.info("Task phase one completed.")

    # Get another reference to the same logger instance
    another_logger_ref = Logger(logger_name="MyAwesomeApp")
    another_logger_ref.info("This log is from another reference, sharing the same logger instance.")

    my_logger.critical("A critical error occurred! The application is about to shut down.")
    my_logger.separator(message="Application End", char="#")
