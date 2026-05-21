from fastapi import HTTPException, status


class AppException(HTTPException):
    """Base class for all application-level exceptions."""


class NotFoundException(AppException):
    def __init__(self, detail: str = "Resource not found."):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class ConflictException(AppException):
    def __init__(self, detail: str = "Resource already exists."):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class UnauthorizedException(AppException):
    def __init__(self, detail: str = "Authentication required."):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class ForbiddenException(AppException):
    def __init__(self, detail: str = "Access denied."):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
