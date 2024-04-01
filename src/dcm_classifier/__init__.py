import importlib.metadata

try:
    # Attempt to retrieve the version of the installed package
    __version__ = importlib.metadata.version("dcm_classifier")
except importlib.metadata.PackageNotFoundError:
    # Fallback version if the package is not installed (e.g., in development)
    __version__ = "0.0.0-dev"
