from typing import List


SYSTEM_INSTRUCTIONS_TXT_OBJ = (
    "You evaluate how well a textual description aligns with VPR-relevant (place-defining) objects detected in an image. "
    "Consider only place-relevant structures/signage (e.g., buildings, bridges, towers, storefronts, signs, street lights, doors/windows, arches, domes, pillars/columns, stairs, fences/railings). "
    "Ignore transient or non-VPR elements such as people, vehicles, animals, sky, grass, and other temporary objects. "
    "Be strict about hallucinations and irrelevant details."
)


def build_text_vs_objects_prompt(objects: List[str], description: str, score_only: bool = False) -> str:
    objects_str = ", ".join(objects) if objects else "(no salient objects detected)"
    if score_only:
        return (
            "Given the detected VPR-relevant objects and the image description, judge their alignment strictly for place recognition.\n"
            "Detected VPR objects: "
            f"{objects_str}\n"
            "Description: "
            f"{description}\n\n"
            "Output only a single integer score 1..5 (5=very consistent; 1=poor)."
        )
    return (
        "Given the detected VPR-relevant objects and the image description, judge their alignment.\n"
        "Detected VPR objects: "
        f"{objects_str}\n"
        "Description: "
        f"{description}\n\n"
        "Provide a brief explanation and a 1..5 score, focusing only on VPR-relevant elements."
    )


