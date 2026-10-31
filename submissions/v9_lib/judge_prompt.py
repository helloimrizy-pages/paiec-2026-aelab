"""Prompt and output schema for the LLM judge. Shared by offline experiments and the submission.

Input mirrors the runtime `predict()` input: a subject dict and an item dict of strings.
The anonymous benchmark_id is not shown to the judge.
"""

from __future__ import annotations

import json
import math

HEAD_CHARS, TAIL_CHARS = 5000, 3000

INSTRUCTIONS = (
    "You are an expert in AI evaluation. You estimate the probability that a specific AI system "
    "produced a response that the benchmark's grader marked as correct on a specific task. "
    "You never see the response itself. Reason about the kind of benchmark the task comes from, "
    "how its grading works, how capable this system was relative to its time, the effect of its "
    "harness and reasoning effort, and how hard this particular task is within its benchmark. "
    "Be calibrated: among tasks you assign probability p, about a fraction p should be correct."
)

SUBJECT_FIELDS = [
    ("normalized_name", "name"),
    ("provider", "provider"),
    ("release_date", "release date"),
    ("access_date", "date the response was recorded"),
    ("harness", "harness / agent scaffold"),
    ("harness_version", "harness version"),
    ("reasoning_effort", "reasoning effort"),
    ("subject_features_extra", "other settings"),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "benchmark_guess": {"type": "string", "description": "Short description of the likely benchmark or task family."},
        "typical_success_rate": {"type": "number", "description": "Typical success rate of this AI system on tasks of this benchmark, 0 to 1."},
        "relative_difficulty": {"type": "string", "enum": ["much_easier", "easier", "typical", "harder", "much_harder"],
                                "description": "This task's difficulty relative to typical tasks of its benchmark."},
        "p_success": {"type": "number", "description": "Probability that this system's response to this task was graded correct, 0 to 1."},
    },
    "required": ["benchmark_guess", "typical_success_rate", "relative_difficulty", "p_success"],
    "additionalProperties": False,
}


def clip_text(s: str) -> str:
    s = s or ""
    if len(s) <= HEAD_CHARS + TAIL_CHARS:
        return s
    return s[:HEAD_CHARS] + f"\n[... {len(s) - HEAD_CHARS - TAIL_CHARS} characters omitted ...]\n" + s[-TAIL_CHARS:]


def _val(d: dict, k: str) -> str:
    v = d.get(k)
    v = "" if v is None else str(v).strip()
    return v if v and v.lower() != "nan" else "not specified"


def user_message(subject: dict, item: dict) -> str:
    lines = ["AI system:"]
    lines += [f"- {label}: {_val(subject, key)}" for key, label in SUBJECT_FIELDS]
    lines += ["", "Task given to the AI system:", "<task>", clip_text(str(item.get("item_content") or "")), "</task>",
              f"Task metadata: {_val(item, 'item_features')}",
              f"Other agents or tools involved: {_val(item, 'interactors')}",
              "", "Estimate the probability that this AI system's response to this task was graded as correct."]
    return "\n".join(lines)


INSTRUCTIONS_V2 = (
    "You are an expert in AI evaluation. You estimate the probability that a specific AI system "
    "produced a response that the benchmark's grader marked as correct on a specific task. "
    "You never see the response itself. Work through these steps before answering:\n"
    "1. Identify the task family and its grading. Decide whether the task is answered in a single "
    "response, or needs multi-step interaction with an environment (a code repository, a browser or "
    "website, a terminal, tools, other agents), where success is checked by executing tests or by "
    "inspecting the final environment state.\n"
    "2. Recall published results for this kind of benchmark: what success rates do AI systems of this "
    "system's generation typically reach? Benchmarks are designed to challenge the systems that are "
    "current when they are released, so typical success rates are often moderate or low, and "
    "execution-graded multi-step tasks are usually much harder for AI systems than they look to humans.\n"
    "3. Adjust for this system's capability relative to its time, its harness, and its reasoning effort.\n"
    "4. Adjust for how hard this particular task is relative to other tasks of the same benchmark.\n"
    "Be calibrated: among tasks you assign probability p, about a fraction p should be correct."
)

SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "benchmark_guess": SCHEMA["properties"]["benchmark_guess"],
        "interaction": {"type": "string", "enum": ["single_response", "multi_step_agentic"],
                        "description": "Single response, or multi-step interaction with an environment."},
        "typical_success_rate": SCHEMA["properties"]["typical_success_rate"],
        "relative_difficulty": SCHEMA["properties"]["relative_difficulty"],
        "p_success": SCHEMA["properties"]["p_success"],
    },
    "required": ["benchmark_guess", "interaction", "typical_success_rate", "relative_difficulty", "p_success"],
    "additionalProperties": False,
}


def request_body(subject: dict, item: dict, model: str, effort: str = "low", max_output_tokens: int = 6000,
                 version: int = 1) -> dict:
    """Body for the OpenAI Responses API with a strict JSON schema (version 1 = submission v2/v3 prompt)."""
    return {
        "model": model,
        "instructions": INSTRUCTIONS if version == 1 else INSTRUCTIONS_V2,
        "input": user_message(subject, item),
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "judgment", "schema": SCHEMA if version == 1 else SCHEMA_V2,
                            "strict": True}},
    }


def parse_output_text(text: str) -> dict:
    """Parse the judge's JSON; raise ValueError if unusable."""
    d = json.loads(text)
    p = float(d["p_success"])
    t = float(d["typical_success_rate"])
    if not (math.isfinite(p) and math.isfinite(t)):
        raise ValueError("non-finite probability")
    d["p_success"] = min(1.0, max(0.0, p))
    d["typical_success_rate"] = min(1.0, max(0.0, t))
    return d


# ---------------------------------------------------------------------------
# In-context judge (version 3): the subject's revealed outcomes on the same benchmark are shown.
# ---------------------------------------------------------------------------

ICL_INSTRUCTIONS = INSTRUCTIONS + (
    " You are also shown other tasks from the same benchmark that this same AI system attempted, with "
    "their graded outcomes. Use them as evidence about how this system performs on this benchmark and on "
    "similar tasks: they set the overall level and reveal which kinds of tasks it tends to fail. They are "
    "different tasks from the target task, so they never reveal its answer."
)
EXCERPT_HEAD, EXCERPT_TAIL = 450, 250


def excerpt(item: dict) -> str:
    s = " ".join(str(item.get("item_content") or "").split())
    if len(s) > EXCERPT_HEAD + EXCERPT_TAIL + 20:
        s = s[:EXCERPT_HEAD] + " [...] " + s[-EXCERPT_TAIL:]
    return f"{s} (metadata: {_val(item, 'item_features')})"


def icl_user_message(subject: dict, item: dict, examples: list) -> str:
    """examples: list of (item_dict, label) revealed for this subject on this benchmark."""
    lines = ["AI system:"]
    lines += [f"- {label}: {_val(subject, key)}" for key, label in SUBJECT_FIELDS]
    lines += ["", f"Graded attempts by this AI system on {len(examples)} other tasks from the same benchmark:"]
    lines += [f"{k + 1}. [{'CORRECT' if y else 'INCORRECT'}] {excerpt(it)}" for k, (it, y) in enumerate(examples)]
    lines += ["", "Target task given to the AI system:", "<task>", clip_text(str(item.get("item_content") or "")), "</task>",
              f"Task metadata: {_val(item, 'item_features')}",
              f"Other agents or tools involved: {_val(item, 'interactors')}",
              "", "Estimate the probability that this AI system's response to the target task was graded as correct."]
    return "\n".join(lines)


def icl_request_body(subject: dict, item: dict, examples: list, model: str, effort: str = "low",
                     max_output_tokens: int = 6000) -> dict:
    return {
        "model": model,
        "instructions": ICL_INSTRUCTIONS,
        "input": icl_user_message(subject, item, examples),
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "judgment", "schema": SCHEMA, "strict": True}},
    }


# ---------------------------------------------------------------------------
# Anchored judge (version 4): reference results from other public benchmarks precede the task.
# ---------------------------------------------------------------------------

ANCHOR_GUIDANCE = (
    " Use the reference results below to calibrate your estimate: they show how often AI systems of "
    "this generation actually succeed on different kinds of benchmark tasks, which is often lower than "
    "the tasks' apparent difficulty suggests. They come from other benchmarks, not from the target task's benchmark."
)


def anchored_request_body(subject: dict, item: dict, anchor_text: str, model: str, effort: str = "low",
                          max_output_tokens: int = 6000) -> dict:
    return {
        "model": model,
        "instructions": INSTRUCTIONS + ANCHOR_GUIDANCE + "\n\n" + anchor_text,
        "input": user_message(subject, item),
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "judgment", "schema": SCHEMA, "strict": True}},
    }
