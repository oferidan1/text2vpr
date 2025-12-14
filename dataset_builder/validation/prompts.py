from typing import List


SYSTEM_INSTRUCTIONS = (
    "You are an evaluator for visual place recognition (VPR). "
    "Given two or more textual descriptions of images from the same cluster, "
    "identify stable shared cues (overlap), and inconsistencies. "
    "Treat permanent structures, spatial layout, building facades, entrances, signage text, "
    "and named points-of-interest as CRITICAL for VPR. "
    "Treat sky/weather/lighting/colors, vegetation state (trees/leaves), transient objects (people, cars, buses), "
    "minor construction, and seasonal changes as NON-CRITICAL. "
    "Non-critical differences must NOT reduce the score and must NOT be cited as reasons for outliers or low scores."
    # TODO: Add a note about the case where only non-critical appearing.
)

# Optional: override system instructions for randomized clusters.
# By default we reuse the main SYSTEM_INSTRUCTIONS. Edit this string if you want
# different behavior specifically for --randomize-clusters.
RANDOM_SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS


def build_pairwise_prompt(desc_a: str, desc_b: str) -> str:
    """
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


# def build_cluster_prompt(descriptions: List[str]) -> str:
#     header = (
#         "Assess cluster-level VPR (Visual Place Recognition) consistency across ALL descriptions.\n"
#         "Only output STRICT JSON with the following keys:\n"
#         '- "score": integer in [1..5] indicating overall cluster consistency for place recognition.\n'
#         '- "rationale": a short one-sentence reason for the assigned score. Be concise but specific — mention concrete details rather than general statements. Do NOT cite non-critical differences (e.g., weather, season, trees/vegetation state, lighting) as justification for lowering the score.\n'
#         '- "outlier_indices": 0-based indices of descriptions that likely do NOT match the main place.\n'
#         '- "outlier_reasons_map": an object mapping each outlier index to a short reason. Clearly explain why each image is an outlier, focusing on specific characteristics related to CRITICAL cues (structures, layout, POIs). Do NOT flag or justify outliers based only on NON-CRITICAL differences (vegetation/season/weather/lighting/transient objects).\n'
#         "The scoring method should involve comparing each pair of descriptions within the cluster and identifying:\n"
#         "- overlap_themes: stable shared cues (structures, layout, or POIs) that appear consistently across many descriptions.\n"
#         "- critical_inconsistencies: contradictions related to permanent structures, layout, or POIs.\n"
#         "- noncritical_inconsistencies: variations related to transient or contextual details (e.g., sky, weather, lighting, color, season, vegetation state). These must NOT lower the score unless they mask or contradict critical cues.\n"
#         "Consider the following when assigning the score:\n"
#         "If too many descriptions are inconsistent or outliers are present, assign a lower score accordingly.\n"
#         "Overall, if it’s plausible that two different descriptions refer to the same place, assign a higher score.\n"
#         "Clusters containing only one description should receive a default score of 5.\n"
#     )
#     numbered = "\n".join([f"{i}. {d}" for i, d in enumerate(descriptions)])
#     return f"{header}\n{numbered}\nJSON:"
# second try
def build_cluster_prompt(descriptions: List[str]) -> str:
    header = (
        "Assess cluster-level VPR (Visual Place Recognition) consistency across ALL descriptions.\n"
        "Only output STRICT JSON with the following keys:\n"
        '- "score": integer in [1..5] indicating overall cluster consistency for place recognition.\n'
        '- "rationale": a short one-sentence reason for the assigned score. Be concise but specific — mention concrete details rather than general statements. Do NOT cite non-critical differences (e.g., weather, season, trees/vegetation state, lighting) as justification for lowering the score.\n'
        '- "outlier_indices": 0-based indices of descriptions that likely do NOT match the main place.\n'
        '- "outlier_reasons_map": an object mapping each outlier index to a short reason. Clearly explain why each image is an outlier, focusing on specific characteristics related to CRITICAL cues (structures, layout, POIs). Do NOT flag or justify outliers based only on NON-CRITICAL differences (vegetation/season/weather/lighting/transient objects).\n'
        "The scoring method should involve comparing each pair of descriptions within the cluster and identifying:\n"
        "- overlap_themes: stable shared cues (structures, layout, or POIs) that appear consistently across many descriptions.\n"
        "- critical_inconsistencies: contradictions related to permanent structures, layout, or POIs.\n"
        "- noncritical_inconsistencies: variations related to transient or contextual details (e.g., sky, weather, lighting, color, season, vegetation state). These must NOT lower the score unless they mask or contradict critical cues.\n"
        "Consider the following when assigning the score:\n"
        "If too many descriptions are inconsistent or outliers are present, assign a lower score accordingly.\n"
        "The golden rule for scoring is: if someone can read the descriptions and confidently assume they refer to the same place, assign a higher score.\n"
        "Clusters containing only one description should receive a default score of 5.\n"
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


# Optional: specialized prompts for randomized clusters mode.
# By default these delegate to the standard cluster/chunk prompts.
# Edit these functions to customize the prompt used when --randomize-clusters is active.
def build_random_cluster_prompt(descriptions: List[str]) -> str:
    return build_cluster_prompt(descriptions)


def build_random_chunk_map_prompt(descriptions: List[str]) -> str:
    return build_chunk_map_prompt(descriptions)


