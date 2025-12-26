"""
AIMO 3 Progress Prize - Source Package

This package contains modules for:
- Model deployment (NIM and UNIM)
- Embedding extraction and visualization
- Performance benchmarking
- Submission generation
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "AIMO 3 Team"

# Package paths
PACKAGE_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = PACKAGE_DIR.parent

# Default paths
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_DATASETS_DIR = PROJECT_DIR / "datasets"
DEFAULT_EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"
DEFAULT_SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

# Module exports
__all__ = [
    "NIMModelDeployer",
    "UNIMModelDeployer", 
    "EmbeddingExtractor",
    "EmbeddingVisualizer",
    "InferenceBenchmark",
    "SubmissionGenerator"
]


def get_version():
    """Return package version."""
    return __version__

