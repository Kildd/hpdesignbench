"""
Parallel structural-artifact library build workflow.

Produces a sealed Arrow IPC library under ``experiment/library/`` with one
artifact per structural ID. Conventional analysis remains the scientific
reference; this workflow only builds the optional library backend.
"""
from __future__ import annotations
