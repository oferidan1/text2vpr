from typing import List


SYSTEM_INSTRUCTIONS = (
    "You are a strict evaluator for visual place recognition (VPR). "
    "Given two short textual descriptions of images from the same cluster, "
    "identify stable shared cues (overlap), and inconsistencies. "
    "Treat permanent structures, spatial layout, building facades, entrances, signage text, "
    "and named points-of-interest as CRITICAL for VPR. "
    "Treat sky/weather/lighting/colors, vegetation state, transient objects (people, cars, buses), "
    "minor construction, and seasonal changes as NON-CRITICAL."
)


def build_pairwise_prompt(desc_a: str, desc_b: str) -> str:
    """
    Ask the model to return STRICT JSON for pairwise VPR consistency.
    Expected JSON fields:
      - overlap: list[str]
      - critical_inconsistencies: list[str]
      - noncritical_inconsistencies: list[str]
      - score: integer in [1..5] (how consistent for place recognition)
      - rationale: short string
    """
    header = (
        "Compare the two descriptions for the SAME physical place (possibly different viewpoint/time).\n"
        "Identify:\n"
        "- overlap: stable, shared cues that strongly suggest same place (structures, layout, POIs).\n"
        "- critical_inconsistencies: contradictions about permanent structures/layout/POIs.\n"
        "- noncritical_inconsistencies: differences about sky/weather/colors/transient/seasonal details.\n"
        "Assign a score 1-5 for VPR consistency (5=highly consistent, 1=conflicting).\n"
        "Return STRICT JSON with keys: overlap, critical_inconsistencies, noncritical_inconsistencies, score, rationale.\n"
        "Descriptions:"
    )
    body = f"A) {desc_a}\nB) {desc_b}\nJSON:"
    return f"{header}\n{body}"


def build_cluster_prompt(descriptions: List[str]) -> str:
    header = (
        "Assess cluster-level VPR consistency across ALL descriptions.\n"
        "Identify:\n"
        "- overlap_themes: stable shared cues (structures/layout/POIs) common across many descriptions.\n"
        "- critical_inconsistencies: contradictions about permanent structures/layout/POIs.\n"
        "- noncritical_inconsistencies: differences about sky/weather/colors/transient/seasonal.\n"
        "- outlier_indices: 0-based indices of descriptions that likely do NOT match the main place.\n"
        "Assign a cluster score 1-5 (5=highly consistent, 1=conflicting).\n"
        "Return STRICT JSON with keys: overlap_themes, critical_inconsistencies, noncritical_inconsistencies, outlier_indices, score, rationale.\n"
        "Descriptions:"
    )
    numbered = "\n".join([f"{i}. {d}" for i, d in enumerate(descriptions)])
    return f"{header}\n{numbered}\nJSON:"


def build_chunk_map_prompt(descriptions: List[str]) -> str:
    header = (
        "Map step: analyze this subset of a cluster.\n"
        "Return STRICT JSON: {overlap_themes, critical_inconsistencies, noncritical_inconsistencies, outlier_indices, score, rationale}.\n"
        "Indices are 0-based within this chunk only.\n"
        "Descriptions:"
    )
    numbered = "\n".join([f"{i}. {d}" for i, d in enumerate(descriptions)])
    return f"{header}\n{numbered}\nJSON:"


