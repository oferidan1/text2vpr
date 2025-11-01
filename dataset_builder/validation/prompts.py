from typing import List


SYSTEM_INSTRUCTIONS = (
    "You are a strict evaluator of textual consistency across multiple short descriptions "
    "that should refer to the same visual place and viewpoint."
)


def build_consistency_prompt(descriptions: List[str]) -> str:
    """
    Build a compact prompt asking the model to assess consistency.
    The model must return a strict JSON with: {"score": 1-5, "explanation": "..."}.
    """
    header = (
        "Assess the INTRA-PLACE textual consistency of these descriptions.\n"
        "- Score 5: all consistent and refer to same place/viewpoint.\n"
        "- Score 1: largely inconsistent or conflicting.\n"
        "Return STRICT JSON: {\"score\": <1-5 integer>, \"explanation\": \"short reason\"}.\n"
        "Descriptions:"\
    )
    numbered = "\n".join([f"{i+1}. {d}" for i, d in enumerate(descriptions)])
    return f"{header}\n{numbered}\nJSON:"


