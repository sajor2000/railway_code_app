"""Retry logic with exponential backoff and circuit breaker."""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Type, Union, Tuple, Set
from datetime import datetime, timedelta

from ..exceptions import (
    APIError, RateLimitError, TimeoutError, MedicalTerminologyError
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise APIError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise APIError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.utcnow() > self.last_failure_time + timedelta(seconds=self.recovery_timeout)
        
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
        
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def exponential_backoff_retry(
    retries: int = 3,
    backoff_in_seconds: float = 1.0,
    max_backoff: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        retries: Maximum number of retry attempts
        backoff_in_seconds: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback function called on each retry
    """
    if retryable_exceptions is None:
        retryable_exceptions = (APIError, TimeoutError, ConnectionError)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == retries:
                        logger.error(f"Max retries ({retries}) reached for {func.__name__}")
                        raise
                    
                    # Handle rate limit errors with specific retry timing
                    if isinstance(e, RateLimitError) and e.details.get('retry_after'):
                        wait_time = e.details['retry_after']
                    else:
                        # Calculate exponential backoff
                        wait_time = min(
                            backoff_in_seconds * (exponential_base ** attempt),
                            max_backoff
                        )
                        
                        # Add jitter if enabled
                        if jitter:
                            wait_time *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{retries} for {func.__name__} "
                        f"after {wait_time:.2f}s delay. Error: {str(e)}"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    time.sleep(wait_time)
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == retries:
                        logger.error(f"Max retries ({retries}) reached for {func.__name__}")
                        raise
                    
                    # Handle rate limit errors with specific retry timing
                    if isinstance(e, RateLimitError) and e.details.get('retry_after'):
                        wait_time = e.details['retry_after']
                    else:
                        # Calculate exponential backoff
                        wait_time = min(
                            backoff_in_seconds * (exponential_base ** attempt),
                            max_backoff
                        )
                        
                        # Add jitter if enabled
                        if jitter:
                            wait_time *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{retries} for {func.__name__} "
                        f"after {wait_time:.2f}s delay. Error: {str(e)}"
                    )
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def retry_with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
    retries: int = 3,
    **retry_kwargs
) -> Callable:
    """
    Combine retry logic with circuit breaker pattern.
    
    Args:
        circuit_breaker: CircuitBreaker instance
        retries: Maximum number of retry attempts
        **retry_kwargs: Additional arguments for exponential_backoff_retry
    """
    def decorator(func: Callable) -> Callable:
        # Apply retry decorator
        retried_func = exponential_backoff_retry(retries=retries, **retry_kwargs)(func)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return circuit_breaker.call(retried_func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await circuit_breaker.async_call(retried_func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Pre-configured retry decorators for common use cases
api_retry = exponential_backoff_retry(
    retries=3,
    backoff_in_seconds=1.0,
    max_backoff=30.0,
    retryable_exceptions=(APIError, TimeoutError, ConnectionError)
)

database_retry = exponential_backoff_retry(
    retries=5,
    backoff_in_seconds=0.5,
    max_backoff=10.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
)

embedding_retry = exponential_backoff_retry(
    retries=3,
    backoff_in_seconds=2.0,
    max_backoff=20.0,
    retryable_exceptions=(APIError, TimeoutError)
)