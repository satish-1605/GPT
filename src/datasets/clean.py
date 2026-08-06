import re
def clean_text(story:str)-> str:
    """
    Preprocess a TinyStories sample.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Normalize line endings to '\n'.
    3. Remove empty samples.
    4. Collapse excessive blank lines.
    """
    if not isinstance(story, str):
        return ""

    
    story = story.strip()

    story = story.replace("\r\n", "\n").replace("\r", "\n")

    if not story:
        return ""

    story = re.sub(r"\n\s*\n+", "\n\n", story)
    return story
