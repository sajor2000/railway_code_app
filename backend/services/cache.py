"""Redis caching service for medical terminology system."""

import asyncio
import hashlib
import json
import pickle
from functools import wraps
from typing import Any, Callable, Optional, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..config import settings
from ..exceptions import CacheError
from ..utils.retry import database_retry

import logging

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service with connection pooling."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[Redis] = None
        self._pool = None
        
    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=True
            )
            self._redis = redis.Redis(connection_pool=self._pool)
            # Test connection
            await self._redis.ping()
            logger.info("Redis cache initialized successfully")
        except RedisError as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise CacheError(f"Redis initialization failed: {e}")
    
    async def close(self):
        """Close Redis connection pool."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
    
    @database_retry
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis:
            return None
            
        try:
            value = await self._redis.get(key)
            if value:
                # Try to deserialize JSON first, then pickle
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return pickle.loads(value.encode('latin-1'))
            return None
        except RedisError as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    @database_retry
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if not self._redis:
            return False
            
        try:
            # Try JSON serialization first, fall back to pickle
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = pickle.dumps(value).decode('latin-1')
            
            ttl = ttl or settings.redis_cache_ttl
            await self._redis.setex(key, ttl, serialized)
            return True
        except RedisError as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    @database_retry
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._redis:
            return False
            
        try:
            result = await self._redis.delete(key)
            return bool(result)
        except RedisError as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    @database_retry
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._redis:
            return False
            
        try:
            return bool(await self._redis.exists(key))
        except RedisError as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    @database_retry
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self._redis:
            return 0
            
        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except RedisError as e:
            logger.error(f"Cache clear pattern error for pattern {pattern}: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._redis:
            return {"status": "disconnected"}
            
        try:
            info = await self._redis.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


# Global cache instance
cache_service = CacheService()


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from function name and arguments."""
    # Create a string representation of arguments
    key_parts = [prefix]
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # For complex objects, use a hash
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")
    
    return ":".join(key_parts)


def cache_result(
    ttl: Optional[int] = None,
    prefix: Optional[str] = None,
    key_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator to cache function results in Redis.
    
    Args:
        ttl: Time to live in seconds (default: from settings)
        prefix: Cache key prefix (default: function name)
        key_func: Custom function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        cache_prefix = prefix or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = generate_cache_key(cache_prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache_service.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_service.set(cache_key, result, ttl)
            logger.debug(f"Cached result for key: {cache_key}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            
            async def _wrapper():
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = generate_cache_key(cache_prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_value = await cache_service.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_value
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                await cache_service.set(cache_key, result, ttl)
                logger.debug(f"Cached result for key: {cache_key}")
                
                return result
            
            return loop.run_until_complete(_wrapper())
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def invalidate_cache(pattern: str) -> Callable:
    """
    Decorator to invalidate cache entries matching a pattern after function execution.
    
    Args:
        pattern: Redis key pattern to invalidate (e.g., "user:*")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            cleared = await cache_service.clear_pattern(pattern)
            logger.debug(f"Invalidated {cleared} cache entries matching pattern: {pattern}")
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            loop = asyncio.get_event_loop()
            cleared = loop.run_until_complete(cache_service.clear_pattern(pattern))
            logger.debug(f"Invalidated {cleared} cache entries matching pattern: {pattern}")
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Pre-configured cache decorators
api_cache = cache_result(ttl=3600, prefix="api")  # 1 hour cache for API responses
embedding_cache = cache_result(ttl=86400, prefix="embedding")  # 24 hour cache for embeddings
search_cache = cache_result(ttl=1800, prefix="search")  # 30 minute cache for searches