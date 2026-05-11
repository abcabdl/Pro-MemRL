import unittest

from proactive_pipeline import (
    aggregate_simulation_votes,
    build_phase_g_conversation,
    build_phase_b_conversation,
    build_phase_c_conversation,
    infer_signals,
    make_tuple_record,
    reaction_to_decision,
    strict_interruption_gate,
)


class TestMentalModelPipeline(unittest.TestCase):
    def test_infer_signals_returns_expected_keys_and_ranges(self) -> None:
        signals = infer_signals(
            [
                {"time": 1, "event": "The user runs pytest and sees a traceback in the terminal."},
                {"time": 2, "event": "The user searches Flask query parameter validation on Google."},
                {"time": 3, "event": "The user edits app.py in Visual Studio Code."},
            ]
        )
        self.assertEqual(
            {"flow", "stuck", "need", "accept", "risk", "uncertainty", "progress", "rejection_memory"},
            set(signals.keys()),
        )
        for value in signals.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_infer_signals_varies_rejection_memory_from_context(self) -> None:
        accepted = infer_signals(
            [
                {"time": 1, "event": "The assistant suggestion was accepted and helped resolve the issue."},
                {"time": 2, "event": "The user continues editing app.py after the help."},
            ]
        )
        rejected = infer_signals(
            [
                {"time": 1, "event": "The assistant suggestion was dismissed as an interruption."},
                {"time": 2, "event": "The user ignores the agent and switches tabs."},
            ]
        )
        self.assertLess(accepted["rejection_memory"], rejected["rejection_memory"])

    def test_vote_aggregation_preserves_majority_and_confidence(self) -> None:
        aggregated = aggregate_simulation_votes(
            [
                {
                    "acceptance": "accept",
                    "acceptance_confidence": 1.0,
                    "flow_impact": "unchanged",
                    "relevance": "highly_relevant",
                    "timing": "good_timing",
                    "reasoning": "vote a",
                },
                {
                    "acceptance": "accept",
                    "acceptance_confidence": 1.0,
                    "flow_impact": "improved",
                    "relevance": "highly_relevant",
                    "timing": "good_timing",
                    "reasoning": "vote b",
                },
                {
                    "acceptance": "ignore",
                    "acceptance_confidence": 0.0,
                    "flow_impact": "unchanged",
                    "relevance": "somewhat_relevant",
                    "timing": "neutral",
                    "reasoning": "vote c",
                },
            ]
        )
        self.assertEqual("accept", aggregated["reaction"]["acceptance"])
        self.assertEqual(0.67, aggregated["reaction"]["acceptance_confidence"])
        self.assertEqual(0.67, aggregated["vote_distributions"]["acceptance"]["accept"])

    def test_negative_reaction_maps_to_silence(self) -> None:
        decision = reaction_to_decision(
            {
                "acceptance": "annoyed",
                "acceptance_confidence": 0.75,
                "flow_impact": "disrupted",
                "relevance": "irrelevant",
                "timing": "bad_timing",
                "reasoning": "bad interruption",
            }
        )
        self.assertFalse(decision["should_intervene"])
        self.assertEqual(0, decision["level"])

    def test_noncode_active_research_allows_light_probe_without_benefit_gate(self) -> None:
        decision = reaction_to_decision(
            {
                "acceptance": "accept",
                "acceptance_confidence": 0.75,
                "flow_impact": "unchanged",
                "relevance": "highly_relevant",
                "timing": "neutral",
                "reasoning": "maybe useful",
            },
            signals={"accept": 0.55, "uncertainty": 0.75},
            observations=[
                {"time": 1, "event": "The user searches Bing for remote work software reviews."},
                {"time": 2, "event": "The user browses article pages and takes notes in a markdown file."},
            ],
        )
        self.assertTrue(decision["should_intervene"])
        self.assertEqual(1, decision["level"])
        self.assertEqual("strict_gate_light_probe_only", decision["reason"])

    def test_code_level2_requires_low_risk_and_low_uncertainty(self) -> None:
        decision = reaction_to_decision(
            {
                "acceptance": "accept",
                "acceptance_confidence": 0.85,
                "flow_impact": "improved",
                "relevance": "highly_relevant",
                "timing": "good_timing",
                "reasoning": "directly useful",
            },
            signals={"accept": 0.45, "uncertainty": 0.75},
            observations=[
                {"time": 1, "event": "The user runs pytest and sees a traceback in the terminal."},
                {"time": 2, "event": "The user reruns pytest and sees the same traceback again."},
            ],
        )
        self.assertTrue(decision["should_intervene"])
        self.assertEqual(1, decision["level"])
        self.assertEqual("helpful_but_risky_or_uncertain", decision["reason"])

    def test_code_level2_when_positive_reaction_and_low_signal_risk(self) -> None:
        decision = reaction_to_decision(
            {
                "acceptance": "accept",
                "acceptance_confidence": 0.85,
                "flow_impact": "improved",
                "relevance": "highly_relevant",
                "timing": "good_timing",
                "reasoning": "directly useful",
            },
            signals={"accept": 0.65, "uncertainty": 0.20, "risk": 0.20},
            observations=[
                {"time": 1, "event": "The user runs pytest and sees a traceback in the terminal."},
                {"time": 2, "event": "The user reruns pytest and sees the same traceback again."},
            ],
        )
        self.assertTrue(decision["should_intervene"])
        self.assertEqual(2, decision["level"])
        self.assertEqual("strict_gate_concrete_blocker", decision["reason"])

    def test_strict_interruption_gate_blocks_strong_flow_without_request(self) -> None:
        gate = strict_interruption_gate(
            observations=[
                {"time": 1, "event": "The user edits app.py in Visual Studio Code."},
                {"time": 2, "event": "The user types implementation code in app.py."},
                {"time": 3, "event": "The user switches between two code tabs in Visual Studio Code."},
            ],
            signals={"flow": 0.85, "progress": 0.70, "need": 0.45, "stuck": 0.20, "risk": 0.20, "rejection_memory": 0.10},
            min_need=0.30,
        )
        self.assertFalse(gate["allow_interruption"])
        self.assertEqual("strong_flow_no_request", gate["reason"])
        self.assertEqual(0.3, gate["gate_thresholds"]["need"])
        self.assertEqual({}, gate["memory_threshold_delta"])

    def test_memory_gate_prior_lowers_need_and_evidence_thresholds_for_missed_help(self) -> None:
        gate = strict_interruption_gate(
            observations=[{"time": 1, "event": "The user sees a traceback and searches debugging documentation."}],
            signals={"flow": 0.55, "need": 0.27, "risk": 0.2},
            gate_prior={
                "recommended_threshold_delta": {"need": -0.08, "evidence": -0.07, "flow": 0.04, "risk": 0.02},
                "recommended_signal_delta": {"need": 0.04, "evidence": 0.03, "flow": 0.0, "risk": 0.0},
            },
        )
        self.assertLess(gate["gate_thresholds"]["need"], 0.3)
        self.assertLess(gate["gate_thresholds"]["evidence"], 0.35)
        self.assertGreater(gate["calibrated_gate_inputs"]["need"], gate["gate_inputs"]["need"])
        self.assertEqual(-0.08, gate["memory_threshold_delta"]["need"])

    def test_memory_gate_prior_raises_need_and_evidence_thresholds_for_rejection_risk(self) -> None:
        gate = strict_interruption_gate(
            observations=[{"time": 1, "event": "The user is browsing search results and switching tabs."}],
            signals={"flow": 0.45, "need": 0.31, "risk": 0.35},
            gate_prior={
                "recommended_threshold_delta": {"need": 0.09, "evidence": 0.08, "flow": -0.02, "risk": -0.04},
                "recommended_signal_delta": {"need": -0.03, "evidence": -0.02, "flow": 0.0, "risk": 0.05},
            },
        )
        self.assertGreater(gate["gate_thresholds"]["need"], 0.3)
        self.assertGreater(gate["gate_thresholds"]["evidence"], 0.35)
        self.assertLess(gate["calibrated_gate_inputs"]["need"], gate["gate_inputs"]["need"])
        self.assertEqual(0.08, gate["memory_threshold_delta"]["evidence"])

    def test_phase_b_record_contains_simulation_target(self) -> None:
        record = make_tuple_record(
            sample_id="1",
            observations=[
                {"time": 1, "event": "The user reads a Flask traceback in the terminal."},
                {"time": 2, "event": "The user reopens app.py in Visual Studio Code."},
            ],
            signals={"need": 0.82, "accept": 0.66, "flow": 0.22, "stuck": 0.61, "risk": 0.32, "uncertainty": 0.18},
            candidate={
                "purpose": "Help resolve a concrete blocker.",
                "proactive_task": "Suggest the missing import fix.",
                "response": "I can suggest the missing import now.",
                "operation": None,
            },
            reaction={
                "acceptance": "accept",
                "acceptance_confidence": 0.67,
                "flow_impact": "improved",
                "relevance": "highly_relevant",
                "timing": "good_timing",
                "reasoning": "This directly helps the blocker.",
            },
            final_output={
                "Purpose": "Help with the failing import.",
                "Thoughts": "The user is blocked on a clear coding issue.",
                "Proactive Task": "Suggest the missing import fix.",
                "Response": "I think the missing import is the next fix to try.",
                "Operation": None,
            },
        )
        conversation = build_phase_b_conversation(record)
        self.assertEqual("system", conversation["conversations"][0]["role"])
        self.assertIn("mental_simulation", conversation["conversations"][1]["content"])
        self.assertIn("simulated_reaction", conversation["conversations"][2]["content"])
        self.assertIn("candidate_intervention", conversation["conversations"][1]["content"])
        self.assertIn("intervention_recommendation", conversation["conversations"][2]["content"])

    def test_phase_b_prefers_fixed_decision_when_present(self) -> None:
        record = make_tuple_record(
            sample_id="1b",
            observations=[
                {"time": 1, "event": "The user searches Java synchronization debugging tips."},
            ],
            signals={"need": 0.82, "accept": 0.66, "flow": 0.22, "stuck": 0.61, "risk": 0.32, "uncertainty": 0.18},
            candidate={
                "purpose": "Help resolve a concrete blocker.",
                "proactive_task": "Suggest the likely synchronization fix.",
                "response": "I can suggest the likely synchronization fix now.",
                "operation": None,
            },
            reaction={
                "acceptance": "dismiss",
                "acceptance_confidence": 0.67,
                "flow_impact": "disrupted",
                "relevance": "irrelevant",
                "timing": "bad_timing",
                "reasoning": "This would interrupt normal work.",
            },
            final_output={
                "Purpose": "Help with the synchronization issue.",
                "Thoughts": "The source label says intervening is appropriate.",
                "Proactive Task": "Suggest the likely synchronization fix.",
                "Response": "Try checking the shared lock scope first.",
                "Operation": None,
            },
            decision={"should_intervene": True, "level": 1, "risk": "medium", "reason": "source_stage2_label"},
        )
        conversation = build_phase_b_conversation(record)
        self.assertIn('"should_intervene": true', conversation["conversations"][2]["content"].lower())
        self.assertIn("source_stage2_label", conversation["conversations"][2]["content"])

    def test_phase_g_record_contains_candidate_target(self) -> None:
        record = {
            "observations": [
                {"time": 1, "event": "The user opens app.py in Visual Studio Code."},
                {"time": 2, "event": "The user reads a traceback in the terminal."},
            ],
            "signals": {
                "need": 0.80,
                "accept": 0.62,
                "flow": 0.18,
                "stuck": 0.58,
                "risk": 0.26,
                "uncertainty": 0.22,
                "progress": 0.14,
                "rejection_memory": 0.11,
            },
            "candidate_intervention": {
                "purpose": "Help resolve the failing run.",
                "proactive_task": "Suggest the likely import fix",
                "response": "I can suggest the likely import fix now.",
                "operation": None,
            },
            "domain": "coding",
            "persona_id": "persona_00",
            "rubric": {"timing": "prefer_brief_help"},
            "history_summary": "Prefers concise help when clearly blocked.",
            "interruption_gate": {"allow_interruption": True, "reason": "gate_pass", "recommended_level": 2},
        }
        conversation = build_phase_g_conversation(record)
        self.assertEqual("system", conversation["conversations"][0]["role"])
        self.assertIn("candidate_generation", conversation["conversations"][1]["content"])
        self.assertIn("\"persona\": \"persona_00\"", conversation["conversations"][1]["content"])
        self.assertIn("\"history_summary\": \"Prefers concise help when clearly blocked.\"", conversation["conversations"][1]["content"])
        self.assertIn("\"interruption_gate\"", conversation["conversations"][1]["content"])
        self.assertIn("\"candidate\"", conversation["conversations"][2]["content"])

    def test_make_tuple_record_uses_reaction_decision_for_phase_c(self) -> None:
        record = make_tuple_record(
            sample_id="2",
            observations=[
                {"time": 1, "event": "The user notices missing burger buns while preparing dinner."},
                {"time": 2, "event": "The user searches nearby delivery options."},
            ],
            signals={"need": 0.70, "accept": 0.50, "flow": 0.20, "stuck": 0.35, "risk": 0.20, "uncertainty": 0.40},
            candidate={
                "purpose": "Handle the missing ingredient problem.",
                "proactive_task": "Search nearby stores for burger buns",
                "response": "I noticed you're missing burger buns. Would you like me to check nearby stores?",
                "operation": None,
            },
            reaction={
                "acceptance": "accept",
                "acceptance_confidence": 1.0,
                "flow_impact": "improved",
                "relevance": "highly_relevant",
                "timing": "good_timing",
                "reasoning": "This would help right now.",
            },
            final_output={
                "Purpose": "Handle the missing ingredient problem.",
                "Thoughts": "The user hit this issue at 18:33:20 and needs help quickly.",
                "Proactive Task": "Search nearby stores for burger buns",
                "Response": "I noticed you're missing burger buns. Would you like me to check nearby stores?",
                "Operation": None,
            },
        )
        self.assertEqual(1, record["final_output"]["Decision"]["level"])
        self.assertNotIn("18:33:20", record["final_output"]["Thoughts"])
        self.assertIn("?", record["final_output"]["Response"])

    def test_make_tuple_record_preserves_explicit_decision_override(self) -> None:
        record = make_tuple_record(
            sample_id="2b",
            observations=[
                {"time": 1, "event": "The user edits a draft report in VS Code."},
            ],
            signals={"need": 0.40, "accept": 0.20, "flow": 0.90, "stuck": 0.10, "risk": 0.20, "uncertainty": 0.30},
            candidate={
                "purpose": "Organize the report.",
                "proactive_task": "Offer to summarize the draft.",
                "response": "I can summarize the draft for you.",
                "operation": None,
            },
            reaction={
                "acceptance": "dismiss",
                "acceptance_confidence": 0.80,
                "flow_impact": "disrupted",
                "relevance": "irrelevant",
                "timing": "bad_timing",
                "reasoning": "The suggestion is distracting.",
            },
            final_output={
                "Purpose": "Support the report drafting task.",
                "Thoughts": "Use the original dataset decision instead of the runtime rule.",
                "Proactive Task": "Offer to summarize the draft.",
                "Response": "I can summarize the draft for you.",
                "Operation": None,
            },
            decision={"should_intervene": True, "level": 1, "risk": "medium", "reason": "source_teacher_decision"},
        )
        self.assertTrue(record["final_output"]["Decision"]["should_intervene"])
        self.assertEqual(1, record["final_output"]["Decision"]["level"])
        self.assertEqual("source_teacher_decision", record["final_output"]["Decision"]["reason"])

    def test_phase_c_conversation_uses_fixed_decision_and_probe_for_level1(self) -> None:
        record = make_tuple_record(
            sample_id="3",
            observations=[
                {"time": 1, "event": "The user fills out a return form."},
                {"time": 2, "event": "The user checks the return status page."},
            ],
            signals={"need": 0.55, "accept": 0.43, "flow": 0.32, "stuck": 0.16, "risk": 0.16, "uncertainty": 0.90},
            candidate={
                "purpose": "Offer help with tracking the return.",
                "proactive_task": "Track the return status",
                "response": "I can track the return status for you.",
                "operation": None,
            },
            reaction={
                "acceptance": "ignore",
                "acceptance_confidence": 0.33,
                "flow_impact": "unchanged",
                "relevance": "somewhat_relevant",
                "timing": "neutral",
                "reasoning": "Maybe later.",
            },
            final_output={
                "Purpose": "Offer help with tracking the return.",
                "Thoughts": "Mild friction but not urgent.",
                "Proactive Task": "Track the return status",
                "Response": "I can track the return status for you.",
                "Operation": None,
            },
        )
        self.assertEqual(0, record["final_output"]["Decision"]["level"])
        self.assertIsNone(record["final_output"]["Response"])
        conversation = build_phase_c_conversation(record)
        self.assertIn("fixed_decision", conversation["conversations"][1]["content"])


if __name__ == "__main__":
    unittest.main()
