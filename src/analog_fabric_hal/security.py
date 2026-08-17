"""Small role-based admission policy for the HAL MVP.

Authorization is performed before a job enters the hardware queue. This makes
the queue a safety boundary as well as a scheduling boundary: unprivileged
requests never become hardware work.

Future work: replace this local identity map with authenticated principals,
resource-level permissions, expiring credentials, and a policy store.
"""
from __future__ import annotations

from enum import Enum

from .analog_fabric import FabricError


class Role(str, Enum):
    INPUT_OUTPUT = "input-output"
    EXPERIMENTER = "experimenter"
    OPERATOR = "operator"
    ADMIN = "admin"


class AuthorizationError(FabricError):
    pass


class AuthorizationPolicy:
    """Maps demo identities to roles and grants the minimum useful actions."""

    _allowed = {
        Role.INPUT_OUTPUT: frozenset({"source", "capture"}),
        Role.EXPERIMENTER: frozenset({"source", "capture", "measure"}),
        Role.OPERATOR: frozenset({"source", "capture", "measure", "program", "power"}),
        Role.ADMIN: frozenset({"source", "capture", "measure", "program", "power", "configure"}),
    }

    def __init__(self, identities: dict[str, Role]) -> None:
        self._identities = dict(identities)

    def require(self, actor: str, action: str) -> None:
        role = self._identities.get(actor)
        if role is None:
            raise AuthorizationError("unknown actor")
        if action not in self._allowed[role]:
            raise AuthorizationError(f"actor role {role.value} cannot perform {action}")
