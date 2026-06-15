import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRESS_RELEASE_FILE = BASE_DIR / "data" / "press_releases.json"


def load_press_releases():
    with open(PRESS_RELEASE_FILE, "r") as file:
        data = json.load(file)

    return pd.DataFrame(data)


KEYWORD_ALIASES = {
    "acquisitions": "acquisition",
    "expansions": "expansion",
    "announcements": "announcement",
}


def _normalize_keyword(keyword: str) -> str:
    keyword = keyword.lower().strip()
    return KEYWORD_ALIASES.get(keyword, keyword.rstrip("s") if keyword.endswith("s") else keyword)


def search_press_releases(keyword):
    df = load_press_releases()

    if not keyword:
        return df

    keyword = _normalize_keyword(keyword)

    mask = (
        df["title"].str.lower().str.contains(keyword, regex=False)
        | df["category"].str.lower().str.contains(keyword, regex=False)
        | df["summary"].str.lower().str.contains(keyword, regex=False)
        | df["insight"].str.lower().str.contains(keyword, regex=False)
    )

    return df[mask]