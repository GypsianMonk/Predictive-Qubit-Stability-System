"""
Predictive Quantum Stability System (PQSS)
===========================================

AI-Driven Predictive Decoherence Prevention for Next-Generation Quantum Computing

This package provides the core framework for predicting qubit decoherence
and implementing preventive interventions before quantum information is lost.
"""

__version__ = "1.0.0"
__author__ = "Research Initiative in Quantum Computing, AI, and Adaptive Control Systems"

from . import telemetry
from . import models
from . import controller
from . import digital_twin
from . import simulator

__all__ = [
    "telemetry",
    "models",
    "controller",
    "digital_twin",
    "simulator",
]
