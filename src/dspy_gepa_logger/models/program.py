"""Program data model for GEPA logging."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgramRecord:
    """Represents a DSPy program configuration.

    Programs are identified by their instruction sets and can be referenced
    by ID across multiple rollouts for normalization.
    """

    program_id: Optional[int] = None  # Set after database insertion
    run_id: str = ""
    signature: Optional[str] = None  # e.g., "question -> answer"
    instructions: dict[str, str] = field(default_factory=dict)  # {component: instruction}
    created_at: Optional[str] = None  # ISO timestamp
    instruction_hash: Optional[str] = None  # SHA256 of instructions for dedup

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "program_id": self.program_id,
            "run_id": self.run_id,
            "signature": self.signature,
            "instructions": self.instructions,
            "created_at": self.created_at,
            "instruction_hash": self.instruction_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgramRecord":
        """Create from dictionary representation."""
        return cls(
            program_id=data.get("program_id"),
            run_id=data.get("run_id", ""),
            signature=data.get("signature"),
            instructions=data.get("instructions", {}),
            created_at=data.get("created_at"),
            instruction_hash=data.get("instruction_hash"),
        )
