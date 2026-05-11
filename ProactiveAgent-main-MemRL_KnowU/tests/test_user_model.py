import unittest
from pathlib import Path

from agent.personalization import PersonaRegistry, UserModel


class TestUserModel(unittest.TestCase):
    def test_update_adjusts_frequency(self) -> None:
        registry = PersonaRegistry.load(Path("dataset/persona_assets"))
        persona = registry.get_persona("persona_00")
        model = UserModel(persona=persona, rubric_by_domain=registry.rubrics["persona_00"])
        initial = model.preferred_frequency
        model.update(
            intervention={"signals": {"flow": 0.2}, "candidate": {"response": "hi"}},
            reaction="annoyed",
            rubric_scores={"personal_preference": 0, "frequency": 0, "timing": 0, "communication": 0},
            domain="coding",
        )
        self.assertLess(model.preferred_frequency, initial)


if __name__ == "__main__":
    unittest.main()
