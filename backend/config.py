"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = "Medical Code Mapper API"
    app_version: str = "2.1.0"
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment name")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")), description="API port (GCP Cloud Run uses PORT env var)")
    api_prefix: str = Field(default="/api", description="API prefix")
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    
    # MongoDB Configuration (optional - used for chat history if provided)
    mongo_url: Optional[str] = Field(default="mongodb://localhost:27017", description="MongoDB connection URL")
    db_name: str = Field(default="medical_coding", description="MongoDB database name")
    
    # Pinecone Configuration
    pinecone_api_key: Optional[SecretStr] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: str = Field(default="medical-biobert", description="Pinecone index name")
    pinecone_dimension: int = Field(default=768, description="BioBERT embedding dimension")
    pinecone_metric: str = Field(default="cosine", description="Distance metric")
    
    # OpenAI Configuration
    openai_api_key: Optional[SecretStr] = Field(default=None, description="OpenAI API key")
    primary_model: str = Field(default="gpt-4o", description="Primary OpenAI model")
    openai_embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    openai_max_tokens: int = Field(default=4096, description="Max tokens")
    openai_temperature: float = Field(default=0.1, description="Temperature")
    
    # Medical API Keys
    umls_api_key: Optional[SecretStr] = Field(default=None, description="UMLS API key")
    umls_username: Optional[str] = Field(default=None, description="UMLS username")
    rxnorm_base_url: Optional[str] = Field(default="https://rxnav.nlm.nih.gov/REST", description="RxNorm API base URL")
    who_icd_client_id: Optional[SecretStr] = Field(default=None, description="WHO ICD API client ID")
    who_icd_client_secret: Optional[SecretStr] = Field(default=None, description="WHO ICD API client secret")
    loinc_username: Optional[str] = Field(default=None, description="LOINC username")
    loinc_password: Optional[SecretStr] = Field(default=None, description="LOINC password")
    snomed_browser_url: Optional[str] = Field(default="https://browser.ihtsdotools.org/snowstorm/snomed-ct", description="SNOMED browser URL")
    
    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    biobert_model_name: str = Field(
        default="dmis-lab/biobert-base-cased-v1.1",
        description="BioBERT model name for medical embeddings"
    )
    embedding_max_length: int = Field(default=512, description="Max sequence length")
    embedding_batch_size: int = Field(default=32, description="Batch size for embeddings")
    
    # Redis Configuration (Railway will provide these via environment variables)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_username: Optional[str] = Field(default=None, description="Redis username")
    redis_password: Optional[SecretStr] = Field(default=None, description="Redis password")
    redis_cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Performance Settings
    max_concurrent_requests: int = Field(default=100, description="Max concurrent requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    batch_size: int = Field(default=100, description="Default batch size")
    max_retries: int = Field(default=3, description="Max retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Search Configuration
    search_top_k: int = Field(default=50, description="Top K results for vector search")
    min_similarity_score: float = Field(default=0.7, description="Minimum similarity score")
    hybrid_alpha: float = Field(default=0.5, description="Hybrid search alpha (0=keyword, 1=vector)")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format"
    )
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    otlp_endpoint: Optional[str] = Field(default=None, description="OTLP endpoint")
    
    # Code Pattern Validation
    code_patterns: Dict[str, str] = Field(
        default={
            "ICD-10-CM": r"^[A-Z]\d{2}\.?\d{0,4}$",
            "SNOMED CT": r"^\d{6,18}$",
            "LOINC": r"^\d{1,5}-\d$",
            "RxNorm": r"^\d+$",
            "CPT": r"^\d{5}$",
            "HCPCS": r"^[A-Z]\d{4}$",
            "NDC": r"^\d{4,5}-\d{3,4}-\d{1,2}$"
        },
        description="Regular expressions for code validation"
    )
    
    # Agent Configuration
    agent_max_iterations: int = Field(default=10, description="Max agent iterations")
    agent_timeout: int = Field(default=60, description="Agent timeout in seconds")
    agent_memory_size: int = Field(default=100, description="Agent memory size")
    
    # Export Configuration
    export_formats: List[str] = Field(
        default=["json", "csv", "excel", "fhir", "omop", "redcap"],
        description="Supported export formats"
    )
    max_export_rows: int = Field(default=10000, description="Max rows for export")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        allowed = {"development", "staging", "production", "testing"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v
    
    @field_validator("hybrid_alpha")
    @classmethod
    def validate_hybrid_alpha(cls, v: float) -> float:
        """Validate hybrid search alpha."""
        if not 0 <= v <= 1:
            raise ValueError("hybrid_alpha must be between 0 and 1")
        return v
    
    @field_validator("min_similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Validate similarity score."""
        if not 0 <= v <= 1:
            raise ValueError("min_similarity_score must be between 0 and 1")
        return v
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            # Redis Cloud format with username
            return f"redis://{self.redis_username}:{self.redis_password.get_secret_value()}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()