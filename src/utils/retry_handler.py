"""
Retry handler and error recovery utilities for TDXAgent.

This module provides comprehensive retry mechanisms and error recovery
strategies for robust operation across various failure scenarios.
"""

import asyncio
import random
from typing import Any, Callable, Optional, Union, Type, List
from functools import wraps
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .logger import TDXLogger


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # "exponential", "linear", "constant"


class RetryableError(Exception):
    """Base class for errors that should trigger retries."""
    pass


class NetworkError(RetryableError):
    """Network-related errors that should be retried."""
    pass


class RateLimitError(RetryableError):
    """Rate limiting errors that should be retried with longer delays."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(Exception):
    """Authentication errors that should not be retried."""
    pass


class RetryHandler:
    """
    Handles retry logic with various backoff strategies.
    
    Features:
    - Exponential, linear, and constant backoff
    - Jitter to prevent thundering herd
    - Configurable retry conditions
    - Detailed logging
    """
    
    def __init__(self, config: RetryConfig = None):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self.logger = TDXLogger.get_logger("tdxagent.retry")
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given retry attempt.
        
        Args:
            attempt: Current retry attempt (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * (attempt + 1)
        else:  # constant
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            attempt: Current retry attempt (0-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if max attempts reached
        if attempt >= self.config.max_retries:
            return False
        
        # Don't retry authentication errors
        if isinstance(exception, AuthenticationError):
            return False
        
        # Retry RetryableError and its subclasses
        if isinstance(exception, RetryableError):
            return True
        
        # Retry common network-related exceptions
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
        
        # Check for common HTTP client exceptions
        try:
            import aiohttp
            retryable_exceptions += (
                aiohttp.ClientError,
                aiohttp.ServerTimeoutError,
                aiohttp.ClientTimeout,
            )
        except ImportError:
            pass
        
        return isinstance(exception, retryable_exceptions)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Retry succeeded after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    self.logger.error(f"Not retrying {type(e).__name__}: {e}")
                    raise
                
                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    
                    # Handle rate limiting specially
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed ({type(e).__name__}: {e}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_exception


def retry_on_failure(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    backoff_strategy: str = "exponential"
):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        backoff_strategy: Backoff strategy ("exponential", "linear", "constant")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                backoff_strategy=backoff_strategy
            )
            
            retry_handler = RetryHandler(config)
            return await retry_handler.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, requests immediately fail
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        
        self.logger = TDXLogger.get_logger("tdxagent.circuit_breaker")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker entering CLOSED state")
    
    def record_failure(self, exception: Exception):
        """Record a failed execution."""
        if isinstance(exception, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.logger.warning("Circuit breaker entering OPEN state from HALF_OPEN")
            elif self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(
                    f"Circuit breaker OPEN: {self.failure_count} failures exceeded "
                    f"threshold of {self.failure_threshold}"
                )
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self.can_execute():
            raise Exception("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.record_success()
            return result
            
        except Exception as e:
            self.record_failure(e)
            raise