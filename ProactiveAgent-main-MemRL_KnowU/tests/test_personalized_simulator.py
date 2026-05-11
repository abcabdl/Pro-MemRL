import unittest
from pathlib import Path

from agent.personalization import PersonaAwareSimulator, PersonaRegistry, UserModel


class TestPersonalizedSimulator(unittest.TestCase):
    def test_simulator_returns_weighted_scores(self) -> None:
        registry = PersonaRegistry.load(Path("dataset/persona_assets"))
        persona = registry.get_persona("persona_00")
        user_model = UserModel(persona=persona, rubric_by_domain=registry.rubrics["persona_00"])
        simulator = PersonaAwareSimulator(registry)
        result = simulator.simulate(
            observations=[{"time": 1.0, "event": "The user sees a traceback in app.py and searches for the fix."}],
            signals={"flow": 0.2, "stuck": 0.8, "need": 0.8, "accept": 0.7, "risk": 0.2, "uncertainty": 0.2},
            domain="coding",
            persona_id="persona_00",
            candidate={"purpose": "Fix a blocker.", "proactive_task": "Suggest the likely import fix", "response": "I can suggest the likely import fix.", "operation": None},
            user_model=user_model,
        )
        self.assertEqual(3, len(result.persona_votes))
        self.assertGreaterEqual(result.aggregated_scores.total_score, 0.0)
        self.assertLessEqual(result.acceptance_confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
