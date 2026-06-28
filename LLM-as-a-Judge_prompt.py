from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=".env")


# =============================================================================
# 1. Judge schema
# =============================================================================

JUDGE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "FBC_SCENARIO_JUDGE",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "situation_understanding": {"type": "number"},
            "component_change_relevance": {"type": "number"},
            "causal_coherence": {"type": "number"},
            "fbc_physical_plausibility": {"type": "number"},
            "disasters_explanation_quality": {"type": "number"},
            "effects_explanation_quality": {"type": "number"},
            "specificity": {"type": "number"},
            "schema_output_consistency": {"type": "number"},
            "ambiguity_penalty": {"type": "number"},
            "invalid_generation": {"type": "boolean"},
            "confidence": {"type": "number"},
            "dominant_interpretation": {"type": "string"},
            "main_strength": {"type": "string"},
            "main_weakness": {"type": "string"},
            "brief_reason": {"type": "string"},
        },
        "required": [
            "situation_understanding",
            "component_change_relevance",
            "causal_coherence",
            "fbc_physical_plausibility",
            "disasters_explanation_quality",
            "effects_explanation_quality",
            "specificity",
            "schema_output_consistency",
            "ambiguity_penalty",
            "invalid_generation",
            "confidence",
            "dominant_interpretation",
            "main_strength",
            "main_weakness",
            "brief_reason",
        ],
    },
}


QUALITY_WEIGHTS = {
    "situation_understanding": 0.14,
    "component_change_relevance": 0.14,
    "causal_coherence": 0.15,
    "fbc_physical_plausibility": 0.15,
    "disasters_explanation_quality": 0.12,
    "effects_explanation_quality": 0.12,
    "specificity": 0.10,
    "schema_output_consistency": 0.08,
    "ambiguity_penalty": -0.12,
}


# =============================================================================
# 2. JSONL / output parsing
# =============================================================================

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            rec["_line_no"] = line_no
            out.append(rec)
        except Exception as e:
            out.append({"_line_no": line_no, "error": f"JSONL parse error: {e}"})
    return out


def extract_json_object(raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if isinstance(raw, dict):
        return raw, ""

    if raw is None:
        return None, "raw is None"

    text = str(raw).strip()
    if not text:
        return None, "empty text"

    lower = text.lower()
    if lower in {"no change", "not possible"}:
        return {"plain_response": text}, ""

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except Exception:
        pass

    text2 = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text2 = re.sub(r"\s*```\s*$", "", text2).strip()
    try:
        obj = json.loads(text2)
        if isinstance(obj, dict):
            return obj, ""
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj, ""
        except Exception as e:
            return None, f"JSON extraction failed: {e}"

    return None, "No JSON object found"


def get_generator_output(rec: Dict[str, Any]) -> Tuple[Any, str]:
    obj = rec.get("final_schema_output")
    if isinstance(obj, dict):
        return obj, ""
    if isinstance(obj, str) and obj.strip():
        parsed, err = extract_json_object(obj)
        if parsed is not None:
            return parsed, err
    return extract_json_object(rec.get("raw_output"))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return default
    except Exception:
        return default


def sanitize_judge_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp numeric scores to expected ranges after JSON schema parsing."""
    out = dict(obj)
    for k in [
        "situation_understanding",
        "component_change_relevance",
        "causal_coherence",
        "fbc_physical_plausibility",
        "disasters_explanation_quality",
        "effects_explanation_quality",
        "specificity",
        "schema_output_consistency",
        "ambiguity_penalty",
    ]:
        out[k] = clamp(safe_float(out.get(k), 0.0), 0.0, 5.0)
    out["confidence"] = clamp(safe_float(out.get("confidence"), 0.0), 0.0, 1.0)
    out["invalid_generation"] = bool(out.get("invalid_generation", False))
    for k in ["dominant_interpretation", "main_strength", "main_weakness", "brief_reason"]:
        out[k] = str(out.get(k, "") or "")
    return out


def scenario_quality(judge_output: Dict[str, Any]) -> float:
    total = 0.0
    for key, weight in QUALITY_WEIGHTS.items():
        total += weight * safe_float(judge_output.get(key), 0.0)
    return clamp(total, 0.0, 5.0)


# =============================================================================
# 3. Optional 10,500-corpus evidence retrieval
# =============================================================================

STOPWORDS = {
    "what", "if", "the", "a", "an", "and", "or", "of", "in", "to", "from",
    "with", "when", "while", "into", "under", "for", "by", "is", "are",
    "be", "being", "been", "system", "boiler", "combustor", "given",
    "larger", "smaller", "usual", "normal",
}

SYNONYMS = {
    "air": ["air", "oxygen", "o2", "inlet", "bottom air", "fluidization"],
    "inlet": ["inlet", "air", "velocity", "flow"],
    "velocity": ["velocity", "flow", "gas velocity", "entrainment"],
    "char": ["char", "fuel", "biomass", "particle", "burnout"],
    "particle": ["particle", "diameter", "size", "solid", "char"],
    "particles": ["particle", "diameter", "size", "solid", "char"],
    "wet": ["wet", "moisture", "water", "temperature", "co"],
    "ash": ["ash", "agglomeration", "slagging", "defluidization"],
    "agglomeration": ["agglomeration", "ash", "sticky", "defluidization"],
    "defluidization": ["defluidization", "fluidization", "bed", "mixing"],
    "outlet": ["outlet", "flue", "pressure", "backpressure"],
    "temperature": ["temperature", "thermal", "heat", "hotspot"],
    "oxygen": ["oxygen", "o2", "air", "combustion", "co"],
    "feed": ["feed", "feeding", "feeder", "fuel", "char"],
}


def tokenize(text: Any) -> List[str]:
    raw = str(text or "").lower()
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", raw)
    out: List[str] = []
    for tok in toks:
        if tok not in STOPWORDS:
            out.append(tok)
        if tok in SYNONYMS:
            out.extend(SYNONYMS[tok])
    return out


def compact(x: Any, n: int = 500) -> str:
    s = re.sub(r"\s+", " ", str(x or "")).strip()
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def flatten_training_item(item: Dict[str, Any]) -> Dict[str, str]:
    inp = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    out = item.get("output", {}) if isinstance(item.get("output"), dict) else {}
    scenario = out.get("scenario", {}) if isinstance(out.get("scenario"), dict) else {}

    chain = scenario.get("causal_chain", [])
    if isinstance(chain, list):
        chain_text = " -> ".join(compact(c, 100) for c in chain[:6])
    else:
        chain_text = compact(chain, 500)

    obs = scenario.get("expected_observable_signals", [])
    if isinstance(obs, list):
        obs_text = "; ".join(compact(o, 100) for o in obs[:6])
    else:
        obs_text = compact(obs, 500)

    search_text = " ".join([
        str(item.get("id", "")),
        str(inp.get("operating_mode", "")),
        str(inp.get("fuel", "")),
        str(inp.get("bed_type", "")),
        str(inp.get("requested_failure_focus", "")),
        str(scenario.get("failure_mode", "")),
        str(scenario.get("initiating_event", "")),
        chain_text,
        obs_text,
    ])

    return {
        "id": str(item.get("id", "")),
        "failure_mode": compact(scenario.get("failure_mode", inp.get("requested_failure_focus", "")), 120),
        "initiating_event": compact(scenario.get("initiating_event", ""), 220),
        "causal_chain": compact(chain_text, 480),
        "observables": compact(obs_text, 300),
        "search_text": search_text,
    }


@lru_cache(maxsize=4)
def load_evidence_cards(evidence_json: str) -> Tuple[Dict[str, str], ...]:
    path = Path(evidence_json)
    if not path.exists():
        return tuple()

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return tuple()

    cards = []
    for it in items:
        if isinstance(it, dict):
            cards.append(flatten_training_item(it))
    return tuple(cards)


def retrieve_evidence(description: str, output_text: str, evidence_json: str, top_k: int) -> str:
    if not evidence_json or top_k <= 0:
        return "No retrieved evidence was provided."

    cards = load_evidence_cards(evidence_json)
    if not cards:
        return "No retrieved evidence was provided."

    query_text = description + " " + output_text
    q_counts = Counter(tokenize(query_text))
    scores: List[Tuple[float, int]] = []

    for i, card in enumerate(cards):
        c_counts = Counter(tokenize(card.get("search_text", "")))
        score = 0.0
        for tok, qn in q_counts.items():
            if tok in c_counts:
                score += min(qn, c_counts[tok]) * (1.0 + min(len(tok), 12) / 12.0)
        if score > 0:
            scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    chunks = []
    for rank, (_, idx) in enumerate(scores[:top_k], start=1):
        card = cards[idx]
        chunks.append(
            f"[Evidence {rank} | {card.get('id')}]\n"
            f"- failure mode: {card.get('failure_mode')}\n"
            f"- initiating event: {card.get('initiating_event')}\n"
            f"- causal pattern: {card.get('causal_chain')}\n"
            f"- observables/effects: {card.get('observables')}\n"
        )

    return "\n".join(chunks) if chunks else "No retrieved evidence was provided."


# =============================================================================
# 4. Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are an LLM-as-a-judge evaluator for generated FBC/MFiX boiler scenario interpretations.

Use deterministic judgement. Apply the rubric consistently.
Do not infer the generator model quality from its name; the model name is not provided.
Do not evaluate actual MFiX simulation execution.
Do not require exact keyword/value correctness unless the output is internally inconsistent.
Evaluate the target response using the original description, retrieved evidence, and the fixed rubric.

Return only the strict JSON object required by the schema.
""".strip()


def build_user_prompt(case_metadata: Dict[str, Any], description: str, target_response: Any, evidence: str) -> str:
    return f"""
# Case Metadata
{json.dumps(case_metadata, ensure_ascii=False, indent=2)}

# Original description
{description}

# Retrieved evidence
The evidence below is optional background from the FBC scenario corpus.
It is not an answer key. Use it only to judge whether the generated disasters/effects are generally consistent with FBC mechanisms.

{evidence}

# Target response to evaluate
{json.dumps(target_response, ensure_ascii=False, indent=2) if isinstance(target_response, dict) else str(target_response)}

# Rubric
Score each item from 0 to 5.

1. situation_understanding
Does the target response reflect the original description?

2. component_change_relevance
Are the selected component/trend/keyword choices relevant to the described situation?

3. causal_coherence
Do the stated disasters/effects show a coherent chain from situation to component change and system consequence?

4. fbc_physical_plausibility
Are the explanations plausible for a fluidized-bed boiler/combustor at a high level?

5. disasters_explanation_quality
Does the disasters field provide realistic causes or occurrence mechanisms?

6. effects_explanation_quality
Does the effects field explain meaningful system effects?

7. specificity
Is the response specific rather than generic?

8. schema_output_consistency
Are fields internally consistent, nonempty, and aligned with the SITUATION_SCHEMA?

9. ambiguity_penalty
Penalty score from 0 to 5. Higher means worse. Penalize vague, generic, or evasive wording.

Set invalid_generation=true if the output is empty, non-FBC, contradictory, schema-broken, or too vague to use.

Also provide:
- confidence between 0 and 1
- dominant_interpretation
- main_strength
- main_weakness
- brief_reason
""".strip()


# =============================================================================
# 5. OpenAI call
# =============================================================================

def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def call_judge(
    client: OpenAI,
    model: str,
    user_prompt: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": JUDGE_JSON_SCHEMA,
        },
    }

    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    resp = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0

    text = resp.choices[0].message.content or "{}"
    obj = sanitize_judge_output(json.loads(text))

    usage = getattr(resp, "usage", None)
    meta = {
        "judge_latency_s": latency,
        "judge_input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "judge_output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "judge_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "raw_response_id": getattr(resp, "id", None),
    }
    return obj, meta


# =============================================================================
# 6. Main judge loop
# =============================================================================

def judge(args: argparse.Namespace) -> None:
    client = make_client()
    rows = read_jsonl(args.generations)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line).get("experiment_id")))
            except Exception:
                pass

    model_filter = set(args.models or [])
    n_new = 0

    with out_path.open("a", encoding="utf-8") as f:
        for rec in rows:
            exp_id = str(rec.get("experiment_id", ""))
            if args.resume and exp_id in completed:
                continue
            if model_filter and str(rec.get("model_name")) not in model_filter:
                continue

            target_response, parse_error = get_generator_output(rec)

            # Do not expose generator model name to the judge prompt.
            case_metadata = {
                "experiment_id": rec.get("experiment_id"),
                "description_id": rec.get("description_id", rec.get("scenario_id")),
                "run_id": rec.get("run_id"),
                "generator_model_hidden": True,
                "parse_error_before_judge": parse_error,
            }

            description = str(rec.get("description", rec.get("prompt_text", "")))
            output_text = json.dumps(target_response, ensure_ascii=False) if isinstance(target_response, dict) else str(target_response)
            evidence = retrieve_evidence(
                description=description,
                output_text=output_text,
                evidence_json=args.evidence_json,
                top_k=args.top_k_evidence,
            )

            user_prompt = build_user_prompt(
                case_metadata=case_metadata,
                description=description,
                target_response=target_response,
                evidence=evidence,
            )

            out_rec = {
                "experiment_id": rec.get("experiment_id"),
                "description_id": rec.get("description_id", rec.get("scenario_id")),
                "scenario_id": rec.get("scenario_id", rec.get("description_id")),
                "description": description,
                "model_name": rec.get("model_name"),
                "run_id": rec.get("run_id"),
                "judge_provider": "openai",
                "judge_model": args.model,
                "judge_settings": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "strict_json_schema": True,
                    "evidence_json": args.evidence_json,
                    "top_k_evidence": args.top_k_evidence,
                },
                "judge_output": None,
                "judge_raw_output": "",
                "scenario_quality": None,
                "error": "",
            }

            try:
                judge_output, meta = call_judge(
                    client=client,
                    model=args.model,
                    user_prompt=user_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                out_rec["judge_output"] = judge_output
                out_rec["judge_raw_output"] = json.dumps(judge_output, ensure_ascii=False)
                out_rec["scenario_quality"] = scenario_quality(judge_output)
                out_rec.update(meta)

            except Exception as e:
                out_rec["error"] = repr(e)

            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1

    print(f"Saved OpenAI GPT-4.1 LLM-as-judge results: {out_path.resolve()} ({n_new} new rows)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenAI GPT-4.1 LLM-as-a-judge for existing-prompt FBC scenario outputs.")
    p.add_argument("--generations", required=True, help="raw_generations.jsonl")
    p.add_argument("--out-jsonl", default="scenario_judgements_openai_gpt41.jsonl")
    p.add_argument("--model", default="gpt-4.1-2025-04-14")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-temperature", action="store_true", help="Omit temperature parameter.")
    p.add_argument("--no-top-p", action="store_true", help="Omit top_p parameter.")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--evidence-json", default="", help="Optional fbc_scenario_generator_training_10500.json for retrieved evidence.")
    p.add_argument("--top-k-evidence", type=int, default=0)
    p.add_argument("--models", nargs="*", default=None, help="Optional exact generator model_name filter.")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.no_temperature:
        args.temperature = None
    if args.no_top_p:
        args.top_p = None
    judge(args)


if __name__ == "__main__":
    main()
