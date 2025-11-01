from typing import List


SYSTEM_INSTRUCTIONS_TXT_OBJ = (
    "You evaluate how well a textual description aligns with concrete objects detected in an image. "
    "Be strict about hallucinations and irrelevant details."
)


def build_text_vs_objects_prompt(objects: List[str], description: str) -> str:
    objects_str = ", ".join(objects) if objects else "(no salient objects detected)"
    return (
        "Given the detected objects and the image description, judge their alignment.\n"
        "Detected objects: "
        f"{objects_str}\n"
        "Description: "
        f"{description}\n\n"
        "Respond with STRICT JSON having keys: \n"
        "- score: integer 1..5 (5=very consistent; 1=poor consistency)\n"
        "- explanation: short reason\n"
        "- omit_list: list of objects/phrases in the description that should be omitted\n"
        "- suggested_description: a concise alternative that removes hallucinations (do not quote long passages)\n"
        "JSON:"
    )


