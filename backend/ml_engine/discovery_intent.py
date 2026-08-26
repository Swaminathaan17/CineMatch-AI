"""Lightweight, dependency-free natural-language movie intent extraction.

This is deliberately local: no paid LLM/API is required to understand common
movie requests. It extracts genres, moods, decades/years, runtime, rating,
'new'/'classic' preferences, exclusions and a possible reference title.
"""
from __future__ import annotations
import re

GENRE_ALIASES = {
    "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science fiction": "Science Fiction",
    "romcom": "Romance", "rom-com": "Romance", "thriller": "Thriller", "thrillers": "Thriller",
    "horror": "Horror", "horrors": "Horror", "comedy": "Comedy", "comedies": "Comedy",
    "drama": "Drama", "crime": "Crime", "mystery": "Mystery", "action": "Action",
    "adventure": "Adventure", "fantasy": "Fantasy", "animation": "Animation", "animated": "Animation",
    "documentary": "Documentary", "romance": "Romance", "romantic": "Romance", "war": "War",
    "western": "Western", "music": "Music", "musical": "Music", "family": "Family", "history": "History",
}
MOOD_TO_GENRES = {
    "dark": ["Thriller", "Crime", "Horror"], "psychological": ["Thriller", "Mystery"],
    "mind-bending": ["Science Fiction", "Mystery"], "mind bending": ["Science Fiction", "Mystery"],
    "twisty": ["Mystery", "Thriller"], "twist": ["Mystery", "Thriller"],
    "scary": ["Horror"], "creepy": ["Horror"], "funny": ["Comedy"], "hilarious": ["Comedy"],
    "lighthearted": ["Comedy", "Romance"], "light-hearted": ["Comedy", "Romance"],
    "heartwarming": ["Drama", "Romance"], "emotional": ["Drama"], "sad": ["Drama"],
    "uplifting": ["Drama", "Comedy"], "romantic": ["Romance"], "epic": ["Adventure", "Fantasy"],
    "intense": ["Thriller", "Action"], "violent": ["Action", "Crime", "Thriller"],
    "feel good": ["Comedy", "Drama"], "feel-good": ["Comedy", "Drama"],
}
TMDB_GENRE_IDS = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80, "Documentary": 99,
    "Drama": 18, "Family": 10751, "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878, "Thriller": 53, "War": 10752, "Western": 37,
}


def _dedupe(items):
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def parse_intent(query: str) -> dict:
    q = " ".join(query.strip().split())
    low = q.lower()
    genres = []
    for alias, genre in GENRE_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", low):
            genres.append(genre)
    for mood, gs in MOOD_TO_GENRES.items():
        if mood in low:
            genres.extend(gs)

    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", q)]
    year = years[0] if years else None
    decade = None
    m = re.search(r"\b(19|20)\d0s\b", low)
    if m:
        decade = int(re.search(r"\d{4}", m.group()).group())

    runtime_max = None
    runtime_min = None
    m = re.search(r"(?:under|less than|below|shorter than)\s*(\d{2,3})\s*(?:min|mins|minutes|minute|m)\b", low)
    if m: runtime_max = int(m.group(1))
    m = re.search(r"(?:over|more than|longer than)\s*(\d{2,3})\s*(?:min|mins|minutes|minute|m)\b", low)
    if m: runtime_min = int(m.group(1))
    m = re.search(r"\b(\d{2,3})\s*(?:minute|min)\s*(?:movie|film)?\b", low)
    if m and runtime_max is None and runtime_min is None:
        runtime_max = int(m.group(1)) + 10
        runtime_min = max(0, int(m.group(1)) - 10)

    min_rating = None
    m = re.search(r"(?:rated|rating|score(?:d)?|imdb|tmdb)\s*(?:of|above|over|at least)?\s*(\d(?:\.\d)?)", low)
    if m: min_rating = float(m.group(1))
    if "highly rated" in low or "well rated" in low or "good ratings" in low: min_rating = max(min_rating or 0, 7.0)

    sort = "relevance"
    if any(x in low for x in ["new release", "new releases", "latest", "recent", "new movie", "new movies"]): sort = "recent"
    elif any(x in low for x in ["trending", "popular", "what's popular", "whats popular"]): sort = "popular"
    elif any(x in low for x in ["highest rated", "best rated", "top rated"]): sort = "rating"
    elif any(x in low for x in ["classic", "classics", "old movie", "older movie"]): sort = "classic"

    reference_title = None
    for pat in [r"(?:like|similar to|something like)\s+[\"']?(.+?)[\"']?(?:\s+but\s+|\s+and\s+|$)", r"(?:movies? like)\s+(.+)$"]:
        m = re.search(pat, q, flags=re.I)
        if m:
            candidate = m.group(1).strip(" \"'")
            if 1 < len(candidate) < 80:
                reference_title = candidate
                break

    exclusions = []
    m = re.search(r"(?:without|no|avoid|not)\s+([\w\s,-]+?)(?:\s+(?:please|thanks|$))", low)
    if m:
        raw = m.group(1).strip(" ,")
        for token in re.split(r"\s*,\s*|\s+and\s+", raw):
            if token in GENRE_ALIASES: exclusions.append(GENRE_ALIASES[token])
            elif token in GENRE_ALIASES.values(): exclusions.append(token)

    themes = []
    theme_words = ["space", "survival", "heist", "revenge", "detective", "murder", "dreams", "dream", "time travel", "zombie", "ghost", "alien", "superhero", "road trip", "coming of age", "school", "wedding"]
    for t in theme_words:
        if t in low: themes.append(t)

    return {
        "query": q,
        "genres": _dedupe(genres),
        "tmdb_genre_ids": [TMDB_GENRE_IDS[g] for g in _dedupe(genres) if g in TMDB_GENRE_IDS],
        "year": year,
        "decade": decade,
        "runtime_min": runtime_min,
        "runtime_max": runtime_max,
        "min_rating": min_rating,
        "sort": sort,
        "reference_title": reference_title,
        "exclude_genres": _dedupe(exclusions),
        "themes": themes,
    }
