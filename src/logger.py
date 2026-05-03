import logging
from pathlib import Path


class ConsoleFormatter(logging.Formatter):
	# Custom console formatter:
	# - INFO/debug messages print as plain text
	# - WARNING prints with a WARNING prefix
	# - ERROR and above keep the ERROR prefix
	# This keeps normal progress logs clean while still making problems obvious.
	def format(self, record):
		message = record.getMessage()
		if record.levelno >= logging.ERROR:
			return f"ERROR: {message}"
		if record.levelno >= logging.WARNING:
			return f"WARNING: {message}"
		return message


def get_logger(name: str = "bis-re") -> logging.Logger:
	# Central logger factory used by the rest of the project.
	# Each module calls get_logger(__name__) so logs are grouped by module name.
	logger = logging.getLogger(name)
	if not logger.handlers:
		# Only configure handlers once per logger name.
		# This prevents duplicate output when modules are imported multiple times.
		logger.setLevel(logging.INFO)
		logger.propagate = False

		# Console handler: writes to the terminal using the custom formatter above.
		console_handler = logging.StreamHandler()
		console_handler.setFormatter(ConsoleFormatter())
		logger.addHandler(console_handler)

		# File handler: saves a timestamped log record to logs/bis-re.log.
		# This is useful for debugging runs after the terminal output is gone.
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
