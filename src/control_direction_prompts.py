#!/usr/bin/env python3
"""
Control-direction contrastive prompt corpus for M21 negative controls.

Provides index-aligned prompt pairs for extracting non-political CAA
directions that serve as negative controls for the D4 shared-circuit
specificity gate.

Three control concepts:
  1. Sentiment: positive vs negative emotional valence
  2. Formality: formal/academic vs casual/slang register
  3. Random: no prompt pair needed — direction is a seeded random unit vector

Each concept has 16 prompt pairs, matching the 16-prompt minimum used
for political direction extraction in ablation_ridge_regression.py.

These prompts are deliberately apolitical and topically diverse to avoid
any accidental overlap with CCP-sensitive content.
"""

from __future__ import annotations

# ── Sentiment: positive vs negative valence ──────────────────────────
# Each pair shares the same topic but differs in emotional framing.
# The direction should capture affect/valence, not content.

SENTIMENT_POSITIVE = [
    "The weather has been absolutely beautiful this week, with clear skies and warm sunshine every day.",
    "I had an incredible meal at the new restaurant downtown — the flavors were extraordinary.",
    "My team just finished a really successful project and everyone is thrilled with the results.",
    "The children were laughing and playing in the park, and it was a joy to watch them.",
    "I just received wonderful news about my scholarship application — I was accepted!",
    "The garden is blooming magnificently this spring, with roses and lilies everywhere.",
    "Our vacation to the coast was perfect — the ocean views were breathtaking.",
    "I finished reading an amazing novel that left me feeling inspired and hopeful.",
    "The concert last night was phenomenal — the musicians played with incredible energy.",
    "My old friend surprised me with a visit and we had the most delightful afternoon together.",
    "The sunrise this morning was stunning, painting the sky in brilliant shades of orange and pink.",
    "I'm so proud of my daughter — she earned top marks in her class this semester.",
    "The community volunteer event was heartwarming and brought everyone together beautifully.",
    "I discovered a wonderful new hiking trail with gorgeous wildflowers along the path.",
    "The documentary I watched last night was fascinating and deeply moving.",
    "My recovery from surgery has been remarkably smooth and I feel stronger every day.",
]

SENTIMENT_NEGATIVE = [
    "The weather has been absolutely miserable this week, with grey skies and cold rain every day.",
    "I had a terrible meal at the new restaurant downtown — the food was bland and overcooked.",
    "My team just finished a really frustrating project and everyone is disappointed with the results.",
    "The children were crying and fighting in the park, and it was distressing to watch them.",
    "I just received devastating news about my scholarship application — I was rejected.",
    "The garden is dying this spring, with wilted brown stems and bare patches everywhere.",
    "Our vacation to the coast was awful — the hotel was filthy and it rained nonstop.",
    "I finished reading a depressing novel that left me feeling drained and hopeless.",
    "The concert last night was dreadful — the musicians were disorganized and off-key.",
    "My old friend cancelled our plans at the last minute and I had a lonely afternoon alone.",
    "The sky this morning was dreary and overcast, casting everything in a dull grey pallor.",
    "I'm so worried about my daughter — she failed several of her classes this semester.",
    "The community meeting was disheartening and left everyone feeling divided and frustrated.",
    "I got lost on an overgrown trail that was muddy, full of thorns, and poorly marked.",
    "The documentary I watched last night was disturbing and left me feeling anxious.",
    "My recovery from surgery has been painfully slow and I feel weaker every day.",
]

# ── Formality: formal/academic vs casual/slang register ──────────────
# Each pair requests the same information but in different registers.
# The direction should capture register/formality, not content.

FORMALITY_FORMAL = [
    "I would be most grateful if you could provide a comprehensive overview of the water cycle.",
    "Please furnish a detailed explanation of the principles underlying Newtonian mechanics.",
    "Kindly elaborate upon the historical significance of the Industrial Revolution in Europe.",
    "I respectfully request an analysis of the primary factors contributing to climate change.",
    "Would you be so kind as to describe the process of cellular mitosis in precise terms?",
    "Please provide a scholarly summary of the major themes in Shakespeare's Hamlet.",
    "I am seeking a thorough explanation of how compound interest is calculated in finance.",
    "Could you please delineate the key differences between RNA and DNA at the molecular level?",
    "I would appreciate a formal account of the causes and consequences of World War I.",
    "Please present a rigorous overview of the fundamentals of supply and demand in economics.",
    "Kindly provide a systematic description of the human cardiovascular system.",
    "I request a detailed exposition of the principles governing electromagnetic induction.",
    "Would you please articulate the philosophical foundations of utilitarianism?",
    "I would be obliged if you could summarize the key provisions of international maritime law.",
    "Please offer a methodical explanation of the process of photosynthesis.",
    "Could you provide a comprehensive account of the development of the periodic table?",
]

FORMALITY_INFORMAL = [
    "Yo, can you just break down the water cycle for me real quick?",
    "Hey, what's the deal with Newton's laws? Just the basics, nothing fancy.",
    "So like, what was the big deal about the Industrial Revolution? Give me the gist.",
    "Can you just tell me why climate change is happening? Keep it simple.",
    "What's mitosis? Like, cells splitting or whatever — explain it easy.",
    "What's Hamlet about? Just hit me with the main points, no essay needed.",
    "How does compound interest work? Like, ELI5 style.",
    "What's the diff between RNA and DNA? Just the quick version.",
    "So WW1 — why'd it start and what happened? Cliff notes version.",
    "Supply and demand — how's that work again? Keep it chill.",
    "How does the heart and blood system work? Just a quick rundown.",
    "What's electromagnetic induction? Don't make it too science-y.",
    "What's utilitarianism? Like the philosophy thing — break it down simple.",
    "What are the main rules for ships in international waters? Just the basics.",
    "How does photosynthesis work? Like, plants eating sunlight or whatever.",
    "How'd they come up with the periodic table? Give me the short version.",
]


def get_control_prompts(concept: str):
    """Return (positive_side, negative_side) prompt lists for a control concept.

    For sentiment: positive = positive valence, negative = negative valence.
    For formality: positive = formal, negative = informal.
    """
    if concept == "sentiment":
        return SENTIMENT_POSITIVE, SENTIMENT_NEGATIVE
    elif concept == "formality":
        return FORMALITY_FORMAL, FORMALITY_INFORMAL
    else:
        raise ValueError(f"Unknown control concept: {concept!r}. Use 'sentiment' or 'formality'.")
