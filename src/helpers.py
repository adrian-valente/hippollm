import nltk
from typing import List


def first_sentence(text: str) -> str:
    """extract first sentence of text"""
    sentences = nltk.sent_tokenize(text)
    return sentences[0].strip() if sentences else text.strip()


def parse_bullet_points(text: str, only_first_bullets: bool = False) -> List[str]:
    """extract items in lines corresponding to bullet points.
    If only_first_bullets is True, keep only the first group of lines starting with bullets"""
    if text.strip().startswith("None"):
        return []
    
    lines = [x.strip() for x in text.split("\n")]
    if only_first_bullets: # Keep only first line and the bullet points following immediately
        kept_lines = [lines[0]]
        i = 1
        while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("*")):
            kept_lines.append(lines[i])
            i += 1
        lines = kept_lines
    bullets = [x[2:].strip() if x.startswith("- ") or x.startswith('*') else x for x in lines]
    bullets = [x for x in bullets if x and not x.startswith("None")]
    return bullets


def itemize_list(items):
    """Make bullet points from list of strings"""
    return "\n".join(["- " + str(x) for x in items])


def choice_selection(answer: str, choices: List[str]) -> str:
    """See if an answer corresponds to one among a list of choices, even if the text contains
    more information."""
    answer = answer.strip().lower()
    if answer.startswith("none"):
        return None
    for choice in choices:
        if answer.startswith(str(choice).lower()):
            return choice
    # Second pass (robustness)
    for choice in choices:
        if str(choice).lower() in answer:
            return choice
    return None