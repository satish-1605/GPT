"""
Fixed prompts for qualitative evaluation.
"""

PROMPTS = [
    "Once upon a time",
    "There was a little",
    "One day",
    "The little girl",
    "The little boy",
    "The cat was",
    "The dog ran",
    "A friendly rabbit",
    "The king lived",
    "In the forest",
]


#ICL test

CLASSIFICATION_DATA = [
    {
        "text": "I love this movie.",
        "label": "positive"
    },
    {
        "text": "This is an amazing experience.",
        "label": "positive"
    },
    {
        "text": "I hate this product.",
        "label": "negative"
    },
    {
        "text": "This was a terrible experience.",
        "label": "negative"
    },
    {
        "text": "The movie was okay.",
        "label": "neutral"
    },
]

PATTERN_DATA = [
    {
        "input": "The cat is an animal.\nThe dog is an animal.\nThe rose is a",
        "target": " plant"
    },
    {
        "input": "The apple is a fruit.\nThe carrot is a vegetable.\nThe banana is a",
        "target": " fruit"
    },
    {
        "input": "Paris is in France.\nTokyo is in Japan.\nBerlin is in",
        "target": " Germany"
    },
    {
        "input": "2 + 2 = 4.\n3 + 3 = 6.\n5 + 5 =",
        "target": " 10"
    },
    {
        "input": "The sun rises in the east.\nThe sun sets in the west.\nThe moon shines at",
        "target": " night"
    },
    {
        "input": "A dog has four legs.\nA bird has two legs.\nA spider has",
        "target": " eight legs"
    },
]

TEXT_ICL_DATA = [
    {
        "input": "The little boy went to the park.\nHe saw a dog.\n",
        "target": "The dog was playing with a ball."
    },
    {
        "input": "The little girl opened the door.\nShe saw a garden.\n",
        "target": "The garden was full of flowers."
    },
    {
        "input": "Tom walked into the forest.\nHe heard a strange sound.\n",
        "target": "He looked around to see where it came from."
    },
    {
        "input": "The sun was setting.\nThe sky became orange.\n",
        "target": "The birds flew back to their nests."
    },
    {
        "input": "Mary found a small box.\nShe opened it carefully.\n",
        "target": "Inside was a beautiful golden key."
    },
    {
        "input": "John went to the river.\nHe saw a small boat.\n",
        "target": "He decided to get into the boat."
    },
]