import logging
from pathlib import Path


class ConsoleFormatter(logging.Formatter):
	def format(self, record):
		message = record.getMessage()
		if record.levelno >= logging.ERROR:
			return f"ERROR: {message}"
		if record.levelno >= logging.WARNING:
			return f"WARNING: {message}"
		return message


def get_logger(name: str = "bis-re") -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		logger.setLevel(logging.INFO)
		logger.propagate = False

		console_handler = logging.StreamHandler()
		console_handler.setFormatter(ConsoleFormatter())
		logger.addHandler(console_handler)

		log_dir = Path("logs")
		log_dir.mkdir(exist_ok=True)
		file_handler = logging.FileHandler(log_dir / "bis-re.log", encoding="utf-8")
		file_formatter = logging.Formatter(
			"%(asctime)s %(levelname)s: %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		file_handler.setFormatter(file_formatter)
		logger.addHandler(file_handler)
	return logger
