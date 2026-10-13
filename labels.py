"""Closed label sets and label definitions shared by every system in this project.

The baseline uses this file now. The multi-agent treatment (M6) must import the
same file, so both conditions see exactly the same labels and definitions.

Sources for the definitions:
- "Affected individuals", "Other useful information" and "Not applicable" follow
  the wording in Alharbi & Lee (2019), Crisis Detection from Arabic Tweets.
- The other definitions are short team-written descriptions based on the label
  names (the paper's longer names are "Donations, assistance and volunteering
  services" and "Sympathy, prayers and emotional support").
"""

RELEVANCE_LABELS = ("Related", "Not related")

INFO_TYPE_LABELS = (
    "Affected individuals",
    "Infrastructure and utilities damage",
    "Donations and volunteering",
    "Caution and advice",
    "Sympathy and emotional support",
    "Other useful information",
    "Not applicable",
)

RELEVANCE_DEFINITIONS = {
    "Related": "The tweet is about the flood event: the flooding itself, its effects, or the response to it.",
    "Not related": "The tweet is not about a flood event.",
}

INFO_TYPE_DEFINITIONS = {
    "Affected individuals": "Reports on affected, dead, missing, trapped, found or displaced people.",
    "Infrastructure and utilities damage": "Damage to or disruption of roads, bridges, buildings, electricity, water or other services.",
    "Donations and volunteering": "Offers or requests of donations, assistance or volunteering services.",
    "Caution and advice": "Warnings, safety advice or instructions to the public.",
    "Sympathy and emotional support": "Prayers, sympathy or emotional support for the people affected.",
    "Other useful information": "Useful information about the emergency that does not fit the categories above.",
    "Not applicable": "The tweet is not related, or it is related but gives no useful information (for example, a personal opinion).",
}

assert set(RELEVANCE_DEFINITIONS) == set(RELEVANCE_LABELS)
assert set(INFO_TYPE_DEFINITIONS) == set(INFO_TYPE_LABELS)


def label_guide() -> str:
    """Text block with the allowed labels and their definitions, used in prompts."""
    lines = ["Relevance labels:"]
    lines += [f"- {name}: {text}" for name, text in RELEVANCE_DEFINITIONS.items()]
    lines.append("Information Type labels:")
    lines += [f"- {name}: {text}" for name, text in INFO_TYPE_DEFINITIONS.items()]
    return "\n".join(lines)
