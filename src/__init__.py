"""
Retail Analytics Big Data System
Main package initialization
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

from . import batch_layer
from . import speed_layer
from . import serving_layer
from . import ingestion
from . import utils

__all__ = [
    'batch_layer',
    'speed_layer',
    'serving_layer',
    'ingestion',
    'utils'
]
