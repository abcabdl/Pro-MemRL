import json
SYSTEM = '''<Task>
Evaluate the task proposed by the proactive assistant as the user.
</Task>

<Rule>
0. Analyze the current observation to understand your current situation and requirements.
1. If the proposed task is `null` (indicating no task is proposed under the current observation), follow these steps:
   - Accept the `null` task if you believe there is no need for a task.
   - Reject the `null` task if you believe a task is needed.
2. Minimize interruptions from the assistant by only accepting tasks that are valuable.
3. Evaluate the current observation and make a judgment on the proposed task accordingly.
</Rule>

<Format>
You should answer with following JSON format:
{
    "thought": "Give your thoughts first, then provide the judgement of the task.",
    "judgement": "accepted or rejected"
}
</Format>'''

def format_reward_instruction(obs:list[dict],pred_task:str) -> list[dict]:
    inst_dict = {
        "Observations (Time Ascending)": obs,
        "Proposed Task": pred_task,
        "Instruction": "Now give your judgement. You should complete the reasoning process in first person."
    }
    
    return [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":json.dumps(inst_dict,sort_keys=False,ensure_ascii=False,indent=4)}
    ]


DECISION_SYSTEM = '''<Task>
Evaluate whether the proactive assistant made the correct intervention decision and, if it intervened, whether the commitment level is appropriate.
</Task>

<Rules>
0. Analyze the observation history first.
1. If no intervention is needed now, the correct commitment level is 0.
2. If intervention is useful but the user is still progressing normally, the task is still somewhat ambiguous, or a direct fix would be premature, the correct commitment level is 1.
3. Use commitment level 2 only when there is a concrete blocker, obvious mistake, explicit error, repeated failed attempt, or one exact missing next step with immediate payoff.
4. Normal searching, reading docs/results, drafting, implementing, testing, or routine tab/file switching usually do not justify level 2 by themselves. If unsure, prefer the lower level.
5. Judge should_intervene first, then judge commitment_level only when intervention is appropriate.
</Rules>

<Format>
Return strict JSON:
{
    "thought": "text",
    "should_judgement": "correct or incorrect",
    "level_judgement": "correct or incorrect or n/a",
    "judgement": "accepted or rejected"
}
</Format>'''


GOLD_DECISION_SYSTEM = '''<Task>
Infer the correct proactive intervention decision from the observation history alone.
</Task>

<Rules>
0. Use the observation history only. Do not assume any assistant prediction.
1. If no timely, valuable intervention is needed now, set gold_should_intervene=0 and gold_commitment_level=0.
2. Use gold_should_intervene=1 and gold_commitment_level=1 for mild but visible friction that a short, low-pressure suggestion could reduce now, such as repeated search-edit-search loops, repeated comparison/copy-paste, repeated small mistakes, or visible coordination overhead.
3. Do not intervene for normal progress alone. Routine searching, reading docs or results, browsing, drafting, implementing, testing, or routine tab/file switching are usually not enough by themselves.
4. If the user has a clear next step and is actively executing it, prefer gold_should_intervene=0.
5. Use gold_commitment_level=2 only when the user appears concretely blocked: explicit error, obvious typo or misuse, repeated failed attempts, or one exact missing next step with an obvious immediate fix.
6. Being able to infer the likely task or provide a relevant snippet is not enough for gold_commitment_level=2. Paste-ready code or prose should count as level 2 only when it resolves a concrete blocker right now, not when it merely speeds up ordinary work.
7. Searches, result-page reading, tutorial browsing, and exploratory scrolling rarely justify gold_commitment_level=2. If unsure between 1 and 2, prefer 1.
8. Treat a short burst of search-result clicks, scrolling, and reading within the same topic as one normal research flow, not repeated new intervention opportunities.
9. Even if one nearby moment could justify light-touch help, later adjacent observations in the same flow should usually return to gold_should_intervene=0 unless there is a new mistake, blocker, or repeated avoidable cost.
10. Judge should_intervene first, then decide commitment_level only when intervention is appropriate.
</Rules>

<Format>
Return strict JSON:
{
    "thought": "text",
    "gold_should_intervene": 0,
    "gold_commitment_level": 0
}
</Format>'''


GOLD_CANDIDATE_SYSTEM = '''<Task>
Generate candidate proactive tasks that the assistant could offer based on the observation history alone.
</Task>

<Rules>
0. Use the observation history only. Do not assume any hidden user request.
1. Generate a candidate only when it is timely, concrete, low-interruption, and likely to create meaningful immediate value in the current moment.
2. Return only timely, concrete, low-interruption help. Prefer an empty list over speculative, generic, repetitive, or merely nice-to-have suggestions.
3. Generate a candidate only when the observations show a timely opportunity to reduce current friction, coordination overhead, uncertainty, or avoidable repeated cost.
4. If the user is actively progressing, generate a candidate only when the help is clearly timely and immediately useful rather than merely plausible or generally relevant.
5. Use commitment_level=1 by default for exploratory, clarifying, or light-touch help. If unsure between 1 and 2, use 1.
6. A candidate that would merely save a small amount of time, offer a weak shortcut, or provide broadly relevant information is not enough unless it creates clear immediate value for the user right now.
7. Use commitment_level=2 when a stronger, more direct intervention is more appropriate than a light-touch probe and is likely to create clear immediate value in the user's current situation.
8. Avoid generating heavy-handed content that mostly replaces the user's ongoing work unless it resolves immediate friction in a way that clearly justifies interruption.
9. During one continuous local flow, avoid generating repeated candidates that restate essentially the same help across adjacent observations unless the situation has meaningfully changed.
10. Avoid near-duplicate candidates. If multiple candidates address the same underlying need, prefer only the single lightest-touch one.
11. Return at most 2 distinct candidates.
</Rules>

<Format>
Return strict JSON:
{
    "thought": "text",
    "candidates": [
        {
            "task": "short task description",
            "purpose": "why this may help",
            "response": "what the assistant would say",
            "commitment_level": 1
        }
    ]
}
</Format>'''


GOLD_CANDIDATE_JUDGE_SYSTEM = '''<Task>
Evaluate a list of candidate proactive tasks from the user's perspective.
</Task>

<Rules>
0. Use the observation history only and judge whether each candidate is worth interrupting the user for right now.
1. Judge whether the candidate is worth interrupting the user for right now, with care to avoid low-value or mistimed interruptions.
2. Default to rejecting. If you are unsure whether the interruption is necessary now, reject the candidate.
3. Reject help that is premature, generic, speculative, redundant, replaceable by the user's current normal progress, or only abstractly useful.
4. Accept a candidate only when it is timely, concrete, low-interruption, and creates clear immediate value in the user's current situation.
5. Accept a level-1 candidate when a light-touch probe, clarification, or modest suggestion is the more appropriate way to help now and is likely to create enough immediate value without overcommitting.
6. Reject a candidate if it would be only marginally useful, weakly relevant, or better deferred to a later moment when interruption cost is lower.
7. Prefer lighter-touch help while the user can still continue productively. If a weaker and a stronger candidate address the same need, usually prefer the weaker one.
8. Accept a level-2 candidate when the stronger intervention is more appropriate than a lighter-touch alternative and is likely to create clearer immediate value in the user's current situation.
9. Reject heavy-handed content that mainly replaces the user's ongoing work unless it resolves immediate friction strongly enough to justify interruption.
10. In a continuous local flow, reject candidates that merely restate help that would have been equally plausible one moment earlier unless the situation has meaningfully changed.
11. When several candidates address the same underlying need, it is common that at most one should be accepted, usually the shortest and lowest-interruption one.
12. Set reject_all=1 when none of the candidates are clearly worth interrupting the user for now.
</Rules>

<Format>
Return strict JSON:
{
    "thought": "text",
    "reject_all": 0,
    "candidate_judgements": [
        {
            "candidate_index": 0,
            "judgement": "accepted or rejected",
            "reason": "short reason"
        }
    ]
}
</Format>'''


PERSONALIZED_RUBRIC_SYSTEM = '''<Task>
Evaluate one proactive intervention from the perspective of a specific user persona.
</Task>

<Rules>
0. Use the provided persona, personalized rubric, history summary, observations, and candidate action together.
1. Score each rubric dimension as 0 or 1.
2. Set total_score as the sum of the four rubric dimensions.
3. Predict acceptance as one of: accept, ignore, dismiss, annoyed.
4. If total_score <= 1, the intervention should usually not happen.
5. If total_score >= 3, the intervention is likely acceptable.
</Rules>

<Format>
Return strict JSON:
{
    "rubric_scores": {
        "personal_preference": 0,
        "frequency": 0,
        "timing": 0,
        "communication": 0
    },
    "total_score": 0,
    "acceptance": "accept",
    "acceptance_confidence": 0.0,
    "reasoning": "text",
    "intervention_recommendation": {
        "should_intervene": false,
        "level": 0,
        "risk": "low | medium | high",
        "adjustment_hint": "text"
    }
}
</Format>'''


def format_decision_reward_instruction(obs: list[dict], prediction: dict) -> list[dict]:
    inst_dict = {
        "Observations (Time Ascending)": obs,
        "Predicted Decision": prediction.get("Decision"),
        "Predicted Purpose": prediction.get("Purpose"),
        "Predicted Task": prediction.get("Proactive Task"),
        "Predicted Response": prediction.get("Response"),
        "Instruction": "Judge whether the intervention decision and commitment level are correct."
    }
    return [
        {"role":"system","content":DECISION_SYSTEM},
        {"role":"user","content":json.dumps(inst_dict,sort_keys=False,ensure_ascii=False,indent=4)}
    ]


def format_gold_decision_instruction(obs: list[dict]) -> list[dict]:
    inst_dict = {
        "Observations (Time Ascending)": obs,
        "Instruction": "Infer the correct intervention decision from the observations only."
    }
    return [
        {"role":"system","content":GOLD_DECISION_SYSTEM},
        {"role":"user","content":json.dumps(inst_dict,sort_keys=False,ensure_ascii=False,indent=4)}
    ]


def format_gold_candidate_generation_instruction(
    obs: list[dict],
    *,
    max_candidates: int,
) -> list[dict]:
    inst_dict = {
        "Observations (Time Ascending)": obs,
        "Max Candidates": int(max_candidates),
        "Instruction": "Generate up to Max Candidates proactive task candidates from the observations only."
    }
    return [
        {"role":"system","content":GOLD_CANDIDATE_SYSTEM},
        {"role":"user","content":json.dumps(inst_dict,sort_keys=False,ensure_ascii=False,indent=4)}
    ]


def format_gold_candidate_judgement_instruction(
    obs: list[dict],
    candidates: list[dict],
) -> list[dict]:
    inst_dict = {
        "Observations (Time Ascending)": obs,
        "Candidate Tasks": candidates,
        "Instruction": "Judge each candidate and set reject_all when none of them should be proposed now."
    }
    return [
        {"role":"system","content":GOLD_CANDIDATE_JUDGE_SYSTEM},
        {"role":"user","content":json.dumps(inst_dict,sort_keys=False,ensure_ascii=False,indent=4)}
    ]


def format_personalized_rubric_instruction(
    *,
    observations: list[dict],
    domain: str,
    persona: dict,
    rubric: dict,
    history_summary: str,
    candidate: dict,
) -> list[dict]:
    inst_dict = {
        "Domain": domain,
        "Persona": persona,
        "Personalized Rubric": rubric,
        "History Summary": history_summary,
        "Observations (Time Ascending)": observations,
        "Candidate Intervention": candidate,
        "Instruction": "Score the candidate with the four-dimension personalized rubric and return strict JSON.",
    }
    return [
        {"role": "system", "content": PERSONALIZED_RUBRIC_SYSTEM},
        {"role": "user", "content": json.dumps(inst_dict, sort_keys=False, ensure_ascii=False, indent=4)},
    ]
