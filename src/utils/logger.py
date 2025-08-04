"""
Comprehensive logging system for TDXAgent.

This module provides a flexible logging system with:
- Console and file output
- Log rotation
- Colored console output
- Structured logging
- Performance monitoring
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import threading
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    COLORS = {
        'DEBUG': 'dim blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold red'
    }
    
    def __init__(self, console: Console):
        super().__init__()
        self.console = console
    
    def format(self, record: logging.LogRecord) -> str:
        # Create colored level name
        level_color = self.COLORS.get(record.levelname, 'white')
        colored_level = Text(f"[{record.levelname:8}]", style=level_color)
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format message
        message = record.getMessage()
        
        # Format logger name
        logger_name = record.name.split('.')[-1] if '.' in record.name else record.name
        
        return f"{timestamp} {colored_level} {logger_name:15} | {message}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = datetime.now().timestamp()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = datetime.now().timestamp() - self.start_time
            if exc_type:
                self.logger.error(f"Failed {self.operation} after {duration:.2f}s: {exc_val}")
            else:
                self.logger.log(self.level, f"Completed {self.operation} in {duration:.2f}s")


class TDXLogger:
    """
    Main logger class for TDXAgent with advanced features.
    
    Features:
    - Multiple output handlers (console, file, JSON)
    - Log rotation
    - Colored console output
    - Performance monitoring
    - Thread-safe operation
    """
    
    _instances: Dict[str, logging.Logger] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_logger(cls, name: str, 
                   log_file: Optional[str] = None,
                   level: str = "INFO",
                   enable_console: bool = True,
                   enable_file: bool = True,
                   enable_json: bool = False,
                   max_file_size: int = 10 * 1024 * 1024,  # 10MB
                   backup_count: int = 5) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON structured logging
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger instance
        """
        with cls._lock:
            if name in cls._instances:
                return cls._instances[name]
            
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, level.upper()))
            
            # Clear any existing handlers
            logger.handlers.clear()
            
            # Console handler with rich formatting
            if enable_console:
                console = Console()
                console_handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_level=True,
                    show_path=False,
                    rich_tracebacks=True
                )
                console_handler.setLevel(getattr(logging, level.upper()))
                logger.addHandler(console_handler)
            
            # File handler with rotation
            if enable_file and log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(getattr(logging, level.upper()))
                logger.addHandler(file_handler)
            
            # JSON handler for structured logging
            if enable_json and log_file:
                json_log_file = str(Path(log_file).with_suffix('.json'))
                json_handler = logging.handlers.RotatingFileHandler(
                    json_log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                json_handler.setFormatter(JSONFormatter())
                json_handler.setLevel(getattr(logging, level.upper()))
                logger.addHandler(json_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
            
            cls._instances[name] = logger
            return logger
    
    @classmethod
    def setup_application_logging(cls, 
                                 data_directory: str = "TDXAgent_Data",
                                 log_level: str = "INFO") -> logging.Logger:
        """
        Set up logging for the main TDXAgent application.
        
        Args:
            data_directory: Base data directory
            log_level: Application log level
            
        Returns:
            Main application logger
        """
        log_dir = Path(data_directory) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main application log
        main_log_file = log_dir / "tdxagent.log"
        
        # Create main logger
        main_logger = cls.get_logger(
            "tdxagent",
            str(main_log_file),
            log_level,
            enable_console=True,
            enable_file=True,
            enable_json=True
        )
        
        # Create specialized loggers
        cls.get_logger(
            "tdxagent.scrapers",
            str(log_dir / "scrapers.log"),
            log_level,
            enable_console=False,
            enable_file=True
        )
        
        cls.get_logger(
            "tdxagent.llm",
            str(log_dir / "llm.log"),
            log_level,
            enable_console=False,
            enable_file=True
        )
        
        cls.get_logger(
            "tdxagent.storage",
            str(log_dir / "storage.log"),
            log_level,
            enable_console=False,
            enable_file=True
        )
        
        main_logger.info("TDXAgent logging system initialized")
        return main_logger
    
    @classmethod
    def performance_monitor(cls, logger: logging.Logger, operation: str) -> PerformanceLogger:
        """
        Create a performance monitoring context manager.
        
        Args:
            logger: Logger to use for performance logs
            operation: Description of the operation being monitored
            
        Returns:
            Performance logger context manager
        """
        return PerformanceLogger(logger, operation)


def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: str = "INFO",
                **kwargs) -> logging.Logger:
    """
    Convenience function to set up a logger.
    
    This is a simplified interface to TDXLogger.get_logger().
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Log level
        **kwargs: Additional arguments passed to TDXLogger.get_logger()
        
    Returns:
        Configured logger instance
    """
    return TDXLogger.get_logger(name, log_file, level, **kwargs)


# Convenience functions for common logging patterns
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator


def log_async_function_call(logger: logging.Logger):
    """Decorator to log async function calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Async {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Async {func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator
