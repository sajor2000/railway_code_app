"""Circuit breaker pattern for resilient external API calls."""

import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: int = 60      # Seconds before trying again
    success_threshold: int = 2      # Successes to close circuit
    timeout: int = 30              # Request timeout in seconds

class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        # Check if circuit should transition from OPEN to HALF_OPEN
        if (self.state == CircuitState.OPEN and 
            time.time() - self.last_failure_time > self.config.recovery_timeout):
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name}: Transitioning to HALF_OPEN")
        
        # Reject calls if circuit is OPEN
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._on_success()
            return result
            
        except Exception as e:
            # Record failure
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name}: Closed after recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self, error: Exception):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Circuit breaker {self.name}: Failure {self.failure_count}/{self.config.failure_threshold} - {error}")
        
        if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and 
            self.failure_count >= self.config.failure_threshold):
            
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker {self.name}: OPENED due to {self.failure_count} failures")
    
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_until_retry": max(0, self.config.recovery_timeout - (time.time() - self.last_failure_time)) if self.state == CircuitState.OPEN else 0
        }

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

# Global circuit breakers for external services
circuit_breakers = {
    "openai": CircuitBreaker("OpenAI", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=120)),
    "umls": CircuitBreaker("UMLS", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)),
    "rxnorm": CircuitBreaker("RxNorm", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)),
    "snomed": CircuitBreaker("SNOMED", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)),
    "pinecone": CircuitBreaker("Pinecone", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=180)),
    "redis": CircuitBreaker("Redis", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30))
}

def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get circuit breaker for a service."""
    return circuit_breakers.get(service_name, CircuitBreaker(service_name))

async def get_all_circuit_breaker_status() -> dict:
    """Get status of all circuit breakers."""
    return {name: cb.get_status() for name, cb in circuit_breakers.items()}