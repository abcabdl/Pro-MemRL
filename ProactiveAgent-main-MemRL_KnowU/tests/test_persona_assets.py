import unittest
from pathlib import Path

from agent.personalization import PERSONA_VECTOR_DIM, PersonaRegistry


class TestPersonaAssets(unittest.TestCase):
    def test_generated_assets_load(self) -> None:
        registry = PersonaRegistry.load(Path("dataset/persona_assets"))
        self.assertEqual(20, len(registry.personas))
        self.assertEqual(20, len(registry.rubrics))
        for persona_id, vector in registry.persona_vectors.items():
            self.assertEqual(PERSONA_VECTOR_DIM, len(vector), msg=persona_id)

    def test_knn_includes_self(self) -> None:
        registry = PersonaRegistry.load(Path("dataset/persona_assets"))
        neighbors = registry.nearest_personas("persona_00", k=3)
        self.assertEqual(3, len(neighbors))
        self.assertIn("persona_00", neighbors)


if __name__ == "__main__":
    unittest.main()
