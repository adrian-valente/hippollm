import nltk

def first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    return sentences[0].strip() if sentences else text.strip()

def parse_bullet_points(text, only_bullets=False):
    if text.strip().startswith("None"):
        return []
    
    lines = [x.strip() for x in text.split("\n")]
    if only_bullets: # Keep only first line and the bullet points following immediately
        kept_lines = [lines[0]]
        i = 1
        while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("*")):
            kept_lines.append(lines[i])
            i += 1
        lines = kept_lines
    bullets = [x[2:].strip() if x.startswith("- ") or x.startswith('*') else x for x in lines]
    bullets = [x for x in bullets if x and not x.startswith("None")]
    return bullets

def join_bullet_points(items):
    return "\n".join(["- " + str(x) for x in items])

def itemize_list(items):
    return "\n".join(["- " + str(x) for x in items])

def choice_selection(answer, choices):
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