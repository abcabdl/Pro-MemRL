from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .types import (
    PERSONA_VECTOR_DIM,
    PersonaProfile,
    PersonalizedRubric,
    load_persona_vectors,
    load_personas,
    load_rubrics,
)


def _default_assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset" / "persona_assets"


def _vector_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    matches = sum(1 for a, b in zip(left, right) if float(a) == float(b))
    return matches / float(len(left))


class PersonaRegistry:
    def __init__(
        self,
        *,
        personas: dict[str, PersonaProfile],
        rubrics: dict[str, dict[str, PersonalizedRubric]],
        persona_vectors: dict[str, list[float]],
    ) -> None:
        self.personas = personas
        self.rubrics = rubrics
        self.persona_vectors = persona_vectors
        for persona_id, persona in self.personas.items():
            vector = self.persona_vectors.get(persona_id, [])
            if vector and len(vector) != PERSONA_VECTOR_DIM:
                raise ValueError(f"Invalid vector length for persona_id={persona_id}")
            if vector and not persona.vector:
                self.personas[persona_id] = replace(persona, vector=vector)

    @classmethod
    def load(cls, assets_dir: str | Path | None = None) -> "PersonaRegistry":
        base = Path(assets_dir) if assets_dir is not None else _default_assets_dir()
        personas = load_personas(base / "personas.json")
        rubrics = load_rubrics(base / "rubrics.json")
        vectors = load_persona_vectors(base / "persona_vectors.json")
        return cls(personas=personas, rubrics=rubrics, persona_vectors=vectors)

    def get_persona(self, persona_id: str) -> PersonaProfile:
        return self.personas[persona_id]

    def get_rubric(self, persona_id: str, domain: str) -> PersonalizedRubric | None:
        return self.rubrics.get(persona_id, {}).get(domain)

    def similarity(self, left_persona_id: str, right_persona_id: str) -> float:
        return _vector_similarity(
            self.persona_vectors.get(left_persona_id, []),
            self.persona_vectors.get(right_persona_id, []),
        )

    def nearest_personas(self, persona_id: str, k: int = 3) -> list[str]:
        if persona_id not in self.personas:
            raise KeyError(f"Unknown persona_id={persona_id}")
        scored: list[tuple[float, str]] = []
        for other_id in self.personas:
            scored.append((self.similarity(persona_id, other_id), other_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        picked = [other_id for _, other_id in scored[: max(1, int(k))]]
        if persona_id not in picked:
            picked = [persona_id] + [other_id for other_id in picked if other_id != persona_id]
            picked = picked[: max(1, int(k))]
        return picked
