import nltk
import os
from pathlib import Path
import re


def is_yes(answer: str) -> bool:
    """Check if an answer is a yes"""
    return answer.strip().lower().startswith('y')


def first_sentence(text: str) -> str:
    """extract first sentence of text"""
    sentences = nltk.sent_tokenize(text)
    return sentences[0].strip() if sentences else text.strip()


def parse_bullet_points(text: str, only_first_bullets: bool = False) -> list[str]:
    """extract items in lines corresponding to bullet points.
    If only_first_bullets is True, keep only the first group of lines starting with bullets"""
    bullets = ["- ", "* ", "• "]
    
    text = text.strip()
    if text.startswith("None"):
        return []
    
    # Remove the first sentence if necessary
    if text.startswith("Here are") or text.startswith("Facts") or text.startswith("Entities"): 
        text = text.split("\n", 1)[1]
        
    # Infer a badly formatted "None"
    if any(sub in text.split("\n", 1)[0].lower()
           for sub in ("no facts", "no entities")):
        return []
    
    lines = [x.strip() for x in text.split("\n")]
    # Keep only first line and the bullet points following immediately
    if only_first_bullets: 
        kept_lines = [lines[0]]
        i = 1
        while ((i < len(lines)) and 
               (
                   any(lines[i].startswith(bullet) for bullet in bullets) or
                   (re.match(r"^(\d)+\.", lines[i]))
                )
              ):
            kept_lines.append(lines[i])
            i += 1
        lines = kept_lines
    
    # Parse bullet points
    extracted = [
        x[2:].strip() if any(x.startswith(bullet) for bullet in bullets)
        else (x.split('.', 1)[1].strip() if re.match(r"^(\d)+\.", x) 
        else x)
        for x in lines
    ]
    extracted = [x for x in extracted if x and not x.startswith("None")]
    return extracted


def itemize_list(items):
    """Make bullet points from list of strings"""
    return "\n".join(["- " + str(x) for x in items])


def choice_selection(answer: str, choices: list[str]) -> str:
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


def getroot() -> os.PathLike:
    return (Path(__file__).parent / '../..').resolve()
