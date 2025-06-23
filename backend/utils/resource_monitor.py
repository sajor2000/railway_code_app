"""Resource monitoring for deployment safety."""

import psutil
import logging
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    memory_percent: float
    memory_available_mb: float
    cpu_percent: float
    disk_percent: float
    disk_free_mb: float

class ResourceMonitor:
    """Monitor system resources and provide warnings."""
    
    def __init__(self):
        self.memory_warning_threshold = 85.0  # %
        self.disk_warning_threshold = 90.0    # %
        self.cpu_warning_threshold = 90.0     # %
        
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_mb = disk.free / (1024 * 1024)
            
            return ResourceMetrics(
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                cpu_percent=cpu_percent,
                disk_percent=disk_percent,
                disk_free_mb=disk_free_mb
            )
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {e}")
            # Return safe defaults
            return ResourceMetrics(0, 1000, 0, 0, 1000)
    
    def check_resources(self) -> Dict[str, str]:
        """Check resource usage and return warnings."""
        metrics = self.get_current_metrics()
        warnings = []
        
        # Memory warnings
        if metrics.memory_percent > self.memory_warning_threshold:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.memory_available_mb < 100:  # Less than 100MB available
            warnings.append(f"Low memory available: {metrics.memory_available_mb:.1f}MB")
        
        # CPU warnings
        if metrics.cpu_percent > self.cpu_warning_threshold:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Disk warnings
        if metrics.disk_percent > self.disk_warning_threshold:
            warnings.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        if metrics.disk_free_mb < 100:  # Less than 100MB free
            warnings.append(f"Low disk space: {metrics.disk_free_mb:.1f}MB free")
        
        return {
            "metrics": {
                "memory_percent": metrics.memory_percent,
                "memory_available_mb": metrics.memory_available_mb,
                "cpu_percent": metrics.cpu_percent,
                "disk_percent": metrics.disk_percent,
                "disk_free_mb": metrics.disk_free_mb
            },
            "warnings": warnings,
            "status": "warning" if warnings else "healthy"
        }
    
    async def log_resource_usage(self):
        """Log current resource usage for monitoring."""
        metrics = self.get_current_metrics()
        
        logger.info(f"Resource usage - Memory: {metrics.memory_percent:.1f}%, "
                   f"CPU: {metrics.cpu_percent:.1f}%, "
                   f"Disk: {metrics.disk_percent:.1f}%")
        
        # Log warnings if thresholds exceeded
        if metrics.memory_percent > self.memory_warning_threshold:
            logger.warning(f"⚠️ High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.cpu_percent > self.cpu_warning_threshold:
            logger.warning(f"⚠️ High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.disk_percent > self.disk_warning_threshold:
            logger.warning(f"⚠️ High disk usage: {metrics.disk_percent:.1f}%")

# Global resource monitor instance
resource_monitor = ResourceMonitor()