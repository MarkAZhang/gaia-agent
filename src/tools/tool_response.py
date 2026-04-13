"""Structured response types for agent tools."""

from dataclasses import dataclass


@dataclass
class ToolSuccess:
    type: str = "success"
    response: str = ""


@dataclass
class ToolError:
    type: str = "error"
    error: str = ""
