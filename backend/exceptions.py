"""Custom exception hierarchy for medical terminology system."""

from typing import Optional, Dict, Any


class MedicalTerminologyError(Exception):
    """Base exception for medical terminology system."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(MedicalTerminologyError):
    """Raised when configuration is invalid or missing."""
    pass


class PineconeError(MedicalTerminologyError):
    """Raised when Pinecone operations fail."""
    pass


class APIError(MedicalTerminologyError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 api_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.api_name = api_name
        if status_code:
            self.details['status_code'] = status_code
        if api_name:
            self.details['api_name'] = api_name


class UMLSAPIError(APIError):
    """Raised when UMLS API operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="UMLS", **kwargs)


class RxNormAPIError(APIError):
    """Raised when RxNorm API operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="RxNorm", **kwargs)


class WHOICDAPIError(APIError):
    """Raised when WHO ICD API operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="WHO_ICD", **kwargs)


class LOINCAPIError(APIError):
    """Raised when LOINC API operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="LOINC", **kwargs)


class SNOMEDAPIError(APIError):
    """Raised when SNOMED API operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, api_name="SNOMED", **kwargs)


class ValidationError(MedicalTerminologyError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        if field:
            self.details['field'] = field
        if value is not None:
            self.details['value'] = value


class CodeValidationError(ValidationError):
    """Raised when medical code validation fails."""
    
    def __init__(self, message: str, code: str, system: str, **kwargs):
        super().__init__(message, field="code", value=code, **kwargs)
        self.details['system'] = system


class EmbeddingError(MedicalTerminologyError):
    """Raised when embedding generation fails."""
    pass


class WorkflowError(MedicalTerminologyError):
    """Raised when workflow execution fails."""
    
    def __init__(self, message: str, step: Optional[str] = None, 
                 state: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        if step:
            self.details['step'] = step
        if state:
            self.details['state'] = state


class AgentError(MedicalTerminologyError):
    """Raised when agent execution fails."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if agent_name:
            self.details['agent_name'] = agent_name


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        if retry_after:
            self.details['retry_after'] = retry_after


class TimeoutError(MedicalTerminologyError):
    """Raised when operation times out."""
    
    def __init__(self, message: str, timeout: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        if timeout:
            self.details['timeout'] = timeout


class CacheError(MedicalTerminologyError):
    """Raised when cache operations fail."""
    pass


class DatabaseError(MedicalTerminologyError):
    """Raised when database operations fail."""
    pass