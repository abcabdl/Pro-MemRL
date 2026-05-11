import unittest
from pathlib import Path

from agent.personalization import PersonaRegistry, PersonalizedPlanner, UserModel


class TestPersonalizedPlanner(unittest.TestCase):
    def test_flow_gate_blocks(self) -> None:
        registry = PersonaRegistry.load(Path("dataset/persona_assets"))
        persona = registry.get_persona("persona_00")
        user_model = UserModel(persona=persona, rubric_by_domain=registry.rubrics["persona_00"])
        planner = PersonalizedPlanner()
        decision = planner.decide(
            signals={"flow": 0.95, "stuck": 0.1},
            simulation_result={"total_score": 4.0, "acceptance_confidence": 0.9},
            user_model=user_model,
            domain="coding",
            observations=[{"time": 1.0, "event": "The user is actively implementing code."}],
            proactive_task="Suggest a direct fix",
        )
        self.assertFalse(decision["should_intervene"])


if __name__ == "__main__":
    unittest.main()
