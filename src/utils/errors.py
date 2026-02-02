"""Custom exception classes for Overheating Classifier."""


class OverheatingClassifierError(Exception):
    """Base exception for all Overheating Classifier errors."""
    pass


class STLLoadError(OverheatingClassifierError):
    """Failed to load or validate STL file."""
    pass


class SlicingError(OverheatingClassifierError):
    """Error during STL slicing."""
    pass


class AnalysisError(OverheatingClassifierError):
    """Error during energy analysis computation."""
    pass


class SessionNotFoundError(OverheatingClassifierError):
    """Session ID not found in registry."""
    pass


class SessionExpiredError(OverheatingClassifierError):
    """Session has expired."""
    pass


class ValidationError(OverheatingClassifierError):
    """Input validation failed."""
    pass
