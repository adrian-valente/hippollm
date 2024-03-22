import nltk

def first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    return sentences[0].strip() if sentences else text.strip()

def parse_bullet_points(text):
    if text.strip().startswith("None"):
        return []
    lines = [x.strip() for x in text.split("\n")]
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