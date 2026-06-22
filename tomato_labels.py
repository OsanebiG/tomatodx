from __future__ import annotations

TOMATO_CLASS_ORDER = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_yellow_leaf_curl_virus",
]

TOMATO_CLASS_DISPLAY_NAMES = {
    "Bacterial_spot": "Bacterial spot",
    "Early_blight": "Early blight",
    "Healthy": "Healthy",
    "Late_blight": "Late blight",
    "Leaf_Mold": "Leaf mold",
    "Septoria_leaf_spot": "Septoria leaf spot",
    "Spider_mites": "Spider mites",
    "Target_Spot": "Target spot",
    "Tomato_mosaic_virus": "Tomato mosaic virus",
    "Tomato_yellow_leaf_curl_virus": "Tomato yellow leaf curl virus",
}

TOMATO_CLASS_DESCRIPTIONS = {
    "Bacterial spot": "Dark lesions on leaves and fruit.",
    "Early blight": "Concentric rings on older leaves.",
    "Healthy": "No disease detected.",
    "Late blight": "Rapidly spreading lesions.",
    "Leaf mold": "Yellow patches with mold growth.",
    "Septoria leaf spot": "Small circular spots with dark borders.",
    "Spider mites": "Fine stippling and webbing caused by mite feeding.",
    "Target spot": "Brown target-like lesions that expand across leaves.",
    "Tomato mosaic virus": "Mottled coloration and distortion.",
    "Tomato yellow leaf curl virus": "Leaf curling and yellowing.",
}

TOMATO_QUICK_ID_GUIDE = [
    {
        "name": "Early blight",
        "clue": "Brown spots with clear concentric rings.",
        "memory": "Think: target with rings.",
    },
    {
        "name": "Late blight",
        "clue": "Water-soaked dark patches that spread fast.",
        "memory": "Think: wet, rotten, fast.",
    },
    {
        "name": "Septoria leaf spot",
        "clue": "Tiny round spots with gray centers and black dots.",
        "memory": "Think: small and pepper-like dots.",
    },
    {
        "name": "Target spot",
        "clue": "Medium round spots with dark borders and lighter centers.",
        "memory": "Think: clean round target.",
    },
    {
        "name": "Bacterial spot",
        "clue": "Small dark greasy spots, often with yellow halos.",
        "memory": "Think: greasy black specks.",
    },
    {
        "name": "Leaf mold",
        "clue": "Yellow patches on top with olive or green mold below.",
        "memory": "Think: top yellow, bottom fuzzy.",
    },
    {
        "name": "Spider mites",
        "clue": "Fine stippling, pale specks, and possible webbing.",
        "memory": "Think: tiny pinprick damage.",
    },
    {
        "name": "Tomato mosaic virus",
        "clue": "Light and dark green mosaic pattern with distortion.",
        "memory": "Think: patchy color pattern.",
    },
    {
        "name": "Tomato yellow leaf curl virus",
        "clue": "Leaves curl upward and turn yellow.",
        "memory": "Think: curled and yellow.",
    },
    {
        "name": "Healthy",
        "clue": "Even green leaf with no obvious spotting.",
        "memory": "Think: normal green leaf.",
    },
]

_ALIASES = {
    # Canonical PlantVillage folder names
    "Bacterial_spot": "Bacterial spot",
    "Early_blight": "Early blight",
    "Healthy": "Healthy",
    "Late_blight": "Late blight",
    "Leaf_Mold": "Leaf mold",
    "Septoria_leaf_spot": "Septoria leaf spot",
    "Spider_mites": "Spider mites",
    "Target_Spot": "Target spot",
    "Tomato_mosaic_virus": "Tomato mosaic virus",
    "Tomato_yellow_leaf_curl_virus": "Tomato yellow leaf curl virus",
    # Official PlantVillage Tomato folder names
    "Tomato___Bacterial_spot": "Bacterial spot",
    "Tomato___Early_blight": "Early blight",
    "Tomato___healthy": "Healthy",
    "Tomato___Late_blight": "Late blight",
    "Tomato___Leaf_Mold": "Leaf mold",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mites",
    "Tomato___Target_Spot": "Target spot",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    # Friendly variants that may appear in checkpoints or logs
    "Bacterial spot": "Bacterial spot",
    "Early blight": "Early blight",
    "Late blight": "Late blight",
    "Leaf mold": "Leaf mold",
    "Septoria leaf spot": "Septoria leaf spot",
    "Spider mites": "Spider mites",
    "Target spot": "Target spot",
    "Tomato mosaic virus": "Tomato mosaic virus",
    "Tomato yellow leaf curl virus": "Tomato yellow leaf curl virus",
}


def normalize_tomato_label(label: str) -> str:
    """
    Convert any known tomato disease label variant into a human-friendly name.

    Falls back to a cleaned-up version of the input if the label is unknown.
    """
    cleaned = label.strip()
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]

    cleaned = cleaned.replace("Tomato___", "")
    cleaned = cleaned.replace("___", "_")
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return label
    return cleaned[:1].upper() + cleaned[1:]


def build_tomato_class_name_map(class_names: list[str]) -> dict[str, str]:
    """
    Build a folder-name -> display-name map for any tomato class names we know.
    Unknown labels are normalized with ``normalize_tomato_label``.
    """
    return {class_name: normalize_tomato_label(class_name) for class_name in class_names}
