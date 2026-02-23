"""Self-enhancement bias detector.

Checks whether a judge model favors responses from its own model family.
Requires metadata.model_a / metadata.model_b on pairs to identify which
model produced each response.
"""

from __future__ import annotations

from judgebench.models import BiasReport, JudgeVerdict, JudgeConfig, LabeledPair


# Known model families and their identifiers
MODEL_FAMILIES = {
    "anthropic": ["claude", "anthropic"],
    "openai": ["gpt", "openai", "o1", "o3"],
    "google": ["gemini", "palm", "google"],
    "meta": ["llama", "meta"],
    "mistral": ["mistral", "mixtral"],
}


def _get_family(model_name: str) -> str | None:
    """Determine model family from model name."""
    model_lower = model_name.lower()
    for family, keywords in MODEL_FAMILIES.items():
        if any(kw in model_lower for kw in keywords):
            return family
    return None


def detect_self_enhancement_bias(
    verdicts: list[JudgeVerdict],
    pairs: list[LabeledPair],
    judge_config: JudgeConfig,
) -> BiasReport:
    """Detect self-enhancement bias.

    Checks if the judge prefers responses from its own model family
    more than human raters do.

    Requires pairs to have metadata with 'model_a' and 'model_b' fields.

    Self-preference rate: how often judge picks its own family's response
    vs how often humans do. Score = excess self-preference.
    """
    judge_family = _get_family(judge_config.model)
    if judge_family is None:
        return BiasReport(
            bias_type="self_enhancement",
            score=0.0,
            details={"note": "could not determine judge model family"},
            flagged=False,
        )

    pair_map = {p.id: p for p in pairs}
    original_verdicts = [v for v in verdicts if v.position == "original"]

    judge_self_picks = 0
    human_self_picks = 0
    relevant_pairs = 0

    for v in original_verdicts:
        if v.pair_id not in pair_map:
            continue
        pair = pair_map[v.pair_id]
        model_a = pair.metadata.get("model_a", "")
        model_b = pair.metadata.get("model_b", "")

        family_a = _get_family(model_a)
        family_b = _get_family(model_b)

        # Only relevant if one response is from judge's family and the other isn't
        if family_a == judge_family and family_b != judge_family:
            relevant_pairs += 1
            if v.judge_label == "A":
                judge_self_picks += 1
            if pair.human_label == "A":
                human_self_picks += 1
        elif family_b == judge_family and family_a != judge_family:
            relevant_pairs += 1
            if v.judge_label == "B":
                judge_self_picks += 1
            if pair.human_label == "B":
                human_self_picks += 1

    if relevant_pairs == 0:
        return BiasReport(
            bias_type="self_enhancement",
            score=0.0,
            details={"note": "no pairs with identifiable model families"},
            flagged=False,
        )

    judge_self_rate = judge_self_picks / relevant_pairs
    human_self_rate = human_self_picks / relevant_pairs

    # Excess self-preference
    excess = max(0.0, judge_self_rate - human_self_rate)
    score = min(1.0, excess * 2.0)  # Scale: 0.5 excess = 1.0 score

    return BiasReport(
        bias_type="self_enhancement",
        score=score,
        details={
            "judge_family": judge_family,
            "judge_self_preference_rate": judge_self_rate,
            "human_self_preference_rate": human_self_rate,
            "excess_self_preference": excess,
            "relevant_pairs": relevant_pairs,
        },
        flagged=score > 0.3,
    )
