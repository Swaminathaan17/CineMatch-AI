"""Lightweight, dependency-free natural-language movie intent extraction.

This is deliberately local: no paid LLM/API is required to understand common
movie requests. It extracts genres, moods, tones, keywords, decades/years,
runtime, rating, 'new'/'classic' preferences, exclusions and a possible
reference title, then reports how confident it is about each interpretation.

Phase 2.4 additions (all forward-compatible with the Phase 4 parser):
  - MOOD_KEYWORDS / TONE_MOODS  -> mood & tone detection layered on top of the
    original GENRE_ALIASES / MOOD_TO_GENRES genre extraction.
  - normalize_query()           -> punctuation / casing / hyphen / synonym
    normalization ("sci-fi"->"science fiction", "rom-com"->"romance comedy").
  - extract_reference_title()   -> standalone, reusable reference-movie
    matcher ("something like Interstellar").
  - extract_keywords()          -> content-word keyword bag for relevance.
  - parse_intent() now returns moods, tone, keywords, normalized query and a
    0..1 confidence score alongside the original fields.
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
    "sitcom": "Comedy", "spy": "Thriller", "noir": "Thriller", "biopic": "History",
    "superhero": "Action", "true crime": "Crime", "crime": "Crime",
}
MOOD_TO_GENRES = {
    "dark": ["Thriller", "Crime", "Horror"], "psychological": ["Thriller", "Mystery"],
    "mind-bending": ["Science Fiction", "Mystery"], "mind bending": ["Science Fiction", "Mystery"],
    "twisty": ["Mystery", "Thriller"], "twist": ["Mystery", "Thriller"],
    "scary": ["Horror"], "creepy": ["Horror"], "haunted": ["Horror"], "spooky": ["Horror"],
    "funny": ["Comedy"], "hilarious": ["Comedy"], "humorous": ["Comedy"], "comedy": ["Comedy"],
    "lighthearted": ["Comedy", "Romance"], "light-hearted": ["Comedy", "Romance"],
    "heartwarming": ["Drama", "Romance"], "emotional": ["Drama"], "sad": ["Drama"],
    "tearjerker": ["Drama", "Romance"], "inspiring": ["Drama"], "hopeful": ["Drama", "Comedy"],
    "uplifting": ["Drama", "Comedy"], "romantic": ["Romance"], "epic": ["Adventure", "Fantasy"],
    "intense": ["Thriller", "Action"], "violent": ["Action", "Crime", "Thriller"],
    "feel good": ["Comedy", "Drama"], "feel-good": ["Comedy", "Drama"],
    "happy": ["Comedy"], "cheerful": ["Comedy", "Fantasy"],
    "cozy": ["Romance", "Comedy", "Drama"], "relaxing": ["Drama", "Romance", "Comedy"],
    "chill": ["Drama", "Comedy"], "easy": ["Comedy", "Romance"],
    "nostalgic": ["Drama", "Adventure"], "adventurous": ["Adventure", "Action", "Fantasy"],
    "suspenseful": ["Thriller", "Mystery"], "gripping": ["Thriller", "Adventure"],
    "action-packed": ["Action", "Adventure"], "exciting": ["Action", "Adventure"],
    "thought-provoking": ["Drama", "Mystery"], "mindless": ["Comedy", "Action"],
    "dystopian": ["Science Fiction", "Thriller"], "post-apocalyptic": ["Science Fiction", "Thriller"],
    "juicy": ["Drama", "Mystery"], "weird": ["Science Fiction", "Comedy"],
    "bizarre": ["Fantasy", "Horror"], "dreamy": ["Fantasy", "Romance"],
    "melancholy": ["Drama"], "bleak": ["Drama", "Thriller"],
    "cerebral": ["Mystery", "Science Fiction"], "whimsical": ["Comedy", "Fantasy"],
    "surreal": ["Science Fiction", "Fantasy"], "chaotic": ["Comedy", "Action"],
    "pulse-pounding": ["Action", "Thriller"],
    "smart": ["Mystery", "Drama"], "witty": ["Comedy"],
    "beautiful": ["Romance", "Drama"], "stunning": ["Science Fiction", "Fantasy"],
    "awe-inspiring": ["Documentary", "Fantasy"], "cathartic": ["Drama"],
    "sexy": ["Romance", "Thriller"], "steamy": ["Romance"],
    "gory": ["Horror"], "brutal": ["Action", "Horror"], "graphic": ["Horror", "Thriller"],
    "chilling": ["Horror", "Thriller"], "haunting": ["Horror", "Drama"],
}
TMDB_GENRE_IDS = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80, "Documentary": 99,
    "Drama": 18, "Family": 10751, "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878, "Thriller": 53, "War": 10752, "Western": 37,
}

# Mood keywords that map to intent descriptions (kept separate from the genre
# aliases so the parser reports both "what you asked for" and "which genres
# that implies"). Word-boundary-matched, so "light" won't fire for "delight".
MOOD_KEYWORDS = {
    "funny": "Funny", "feel good": "Feel-good", "feel-good": "Feel-good",
    "happy": "Happy", "emotional": "Emotional", "sad": "Sad",
    "dark": "Dark", "scary": "Scary", "relaxing": "Relaxing", "intense": "Intense",
    "uplifting": "Uplifting", "romantic": "Romantic", "inspiring": "Inspiring",
    "mind-bending": "Mind-bending", "mind bending": "Mind-bending",
    "adventurous": "Adventurous", "nostalgic": "Nostalgic", "suspenseful": "Suspenseful",
    "cozy": "Cozy", "light": "Light", "lighthearted": "Lighthearted",
    "serious": "Serious", "disturbing": "Disturbing", "thought-provoking": "Thought-provoking",
    "psychological": "Psychological", "chilling": "Chilling", "spooky": "Spooky",
    "creepy": "Creepy", "hilarious": "Hilarious", "heartwarming": "Heartwarming",
    "epic": "Epic", "action-packed": "Action-packed", "gripping": "Gripping",
    "weird": "Weird", "witty": "Witty", "whimsical": "Whimsical", "exciting": "Exciting",
}

# Tone adjectives: these are about the overall feel, not a single genre.
# "Serious"/"Light" etc. pair with but do not replace the mood mapping.
TONE_MOODS = {
    "light": "Light", "lighthearted": "Lighthearted", "feel good": "Feel-good",
    "feel-good": "Feel-good", "happy": "Happy", "cozy": "Cozy", "romantic": "Romantic",
    "uplifting": "Uplifting", "cheerful": "Cheerful", "funny": "Funny",
    "silly": "Silly", "easy": "Easygoing", "chill": "Laid-back",
    "dark": "Dark", "serious": "Serious", "disturbing": "Disturbing", "bleak": "Bleak",
    "intense": "Intense", "suspenseful": "Suspenseful", "moody": "Moody",
    "emotional": "Emotional", "melancholy": "Melancholy", "sad": "Sad", "inspiring": "Inspiring",
    "nostalgic": "Nostalgic", "dreamy": "Dreamy", "surreal": "Surreal", "weird": "Weird",
    "gripping": "Gripping", "epic": "Epic", "mind-bending": "Mind-bending",
    "mind bending": "Mind-bending", "adventurous": "Adventurous",
    "thought-provoking": "Thought-provoking", "cerebral": "Cerebral", "cathartic": "Cathartic",
}

# Small textual normalizations applied before matching. These are applied from
# longest to shortest so "rom-com" -> "romance comedy" happens before the
# genre-alias matching sees it.
NORMALIZE_SYNONYMS = {
    "sci-fi": "science fiction", "scifi": "science fiction", "rom-com": "romance comedy",
    "romcom": "romance comedy", "feel-good": "feel good", "light-hearted": "lighthearted",
    "action-packed": "action packed", "mind-bending": "mind bending",
    "sit-com": "sitcom", "true-life": "true life", "underrated": "",
}

# Stop words / request scaffolding removed from the keyword bag.
_KEYWORD_STOP = {
    "the", "a", "an", "and", "or", "but", "for", "with", "like", "want", "from", "that",
    "this", "some", "about", "give", "gimme", "please", "thanks", "thank", "recommend",
    "suggest", "need", "would", "there", "movie", "movies", "film", "films", "something",
    "tonight", "today", "now", "what", "show", "watch", "watching", "really",
    "very", "too", "more", "less", "not", "no", "without", "avoid", "i", "me", "my", "in",
    "of", "to", "it", "on", "at", "new", "old", "classic", "recent", "good", "great",
    "best", "top", "wanna", "gotta", "kinda", "sorta", "after", "little",
}

_MULTI_WORD_KEYWORDS = [
    "time travel", "science fiction", "road trip", "serial killer", "true story",
    "coming of age", "feel good", "space opera", "post apocalyptic", "mind bending",
]

_THEME_WORDS = [
    "space", "survival", "heist", "revenge", "detective", "murder", "dreams", "dream",
    "time travel", "zombie", "ghost", "alien", "superhero", "road trip", "coming of age",
    "school", "wedding", "vampire", "werewolf", "pirate", "ninja", "robot", "dinosaur",
    "virus", "pandemic", "serial killer", "conspiracy", "magic", "wizard", "dragon",
]

_KNOWN_GENRE_STRINGS = sorted(
    {"Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
     "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery",
     "Romance", "Science Fiction", "Thriller", "War", "Western"},
    key=len, reverse=True,
)


def _dedupe(items):
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def normalize_query(query: str) -> str:
    """Normalize casing, punctuation, repeated whitespace and common synonyms
    ("sci-fi" -> "science fiction", "rom-com" -> "romance comedy"). Kept
    deterministic and idempotent. Never exposed as-is in user-facing
    explanations."""
    q = re.sub(r"[?!.,;]+", " ", query).strip()
    q = re.sub(r"\s+", " ", q)
    low = q.lower()
    for alias, replacement in sorted(NORMALIZE_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if replacement:
            low = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", replacement, low)
        else:
            low = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def extract_reference_title(query: str):
    """Return the reference movie title if the query names one, else None.
    Standalone and reusable; handles quoted and unquoted titles."""
    patterns = [
        r"(?i)(?:something\s+)?(?:like|similar\s+to|movies?\s+like|films?\s+like)\s+[\"']?(.+?)[\"']?(?=\s+but\s+|\s+and\s+|\s*$)",
        r"(?i)in\s+the\s+style\s+of\s+[\"']?(.+?)[\"']?(?=\s+but\s+|\s+and\s+|\s*$)",
        r"(?i)if\s+(?:i\s+)?(?:loved|liked|enjoyed)\s+[\"']?(.+?)[\"']?(?=\s+(?:then|please|$))",
        r"(?i)(?:reminds?\s+me\s+of)\s+[\"']?(.+?)[\"']?(?=\s+but\s+|\s+and\s+|\s*$)",
    ]
    for pat in patterns:
        m = re.search(pat, query)
        if m:
            candidate = m.group(1).strip(" \"'")
            if 1 < len(candidate) < 80:
                return candidate
    return None


def extract_keywords(query: str, exclude: str | None = None) -> list[str]:
    """Content-word keywords from the query. Single words minus stop words,
    plus a small set of multi-word phrases ("time travel", "science fiction")
    kept whole so they can match overviews directly.

    `exclude` (typically a reference movie title) is masked out first so words
    inside a title are never fed back as independent request keywords.
    """
    low = normalize_query(query).lower()
    if exclude:
        low = _mask_reference_title(low, exclude)
    words = [
        w.strip("'")
        for w in re.findall(r"[a-z0-9']+", low)
        if len(w.strip("'")) > 2 and w.strip("'") not in _KEYWORD_STOP
    ]
    for phrase in _MULTI_WORD_KEYWORDS:
        if phrase in low:
            words.append(phrase)
    return sorted(set(words))


def _genre_mentions(low: str) -> list[str]:
    found = []
    for g in _KNOWN_GENRE_STRINGS:
        if re.search(r"(?<!\w)" + re.escape(g.lower()) + r"(?!\w)", low):
            found.append(g)
    return found


def _matches(low: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", low) is not None


def _mask_reference_title(low: str, reference_title: str | None) -> str:
    """Remove one reference-title span from a normalized query so words inside
    a movie title (e.g. "dark" in "The Dark Knight") are never re-read as
    independent mood/tone/genre keywords the user asked for. Deterministic and
    local: everything else in the query stays untouched, so a genuine mood
    that appears *outside* the title is still detected."""
    if not reference_title:
        return low
    ref = normalize_query(reference_title)
    if len(ref) < 2 or ref == low:
        return low
    masked = re.sub(r"(?<!\w)" + re.escape(ref) + r"(?!\w)", " ", low, count=1)
    return re.sub(r"\s+", " ", masked).strip()


def parse_intent(query: str) -> dict:
    """Extract structured recommendation intent from a natural-language query.

    Returns:
        query, normalized_query, genres, tmdb_genre_ids, moods, mood_genres,
        tone, keywords, reference_title, year, decade, runtime_min, runtime_max,
        min_rating, sort, exclude_genres, themes, confidence
    """
    q = " ".join(query.strip().split())
    low = normalize_query(q)

    # Reference titles must be resolved BEFORE mood/tone/genre matching so
    # words inside the title are not interpreted as independent request
    # keywords ("movies like The Dark Knight" must not detect a "dark" mood).
    reference_title = extract_reference_title(q)
    match_text = _mask_reference_title(low, reference_title)

    genres = []
    for alias, genre in GENRE_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", match_text):
            genres.append(genre)
    for mood, gs in MOOD_TO_GENRES.items():
        if _matches(match_text, mood):
            genres.extend(gs)
    genres += _genre_mentions(match_text)

    moods = []
    for kw, label in MOOD_KEYWORDS.items():
        if _matches(match_text, kw):
            moods.append(label)
    # "but" contrasts with the main request; a "but not"-style exclusion is
    # handled below so "something scary but not too gory" stays Scary + Horror.
    mood_genres = []
    for m in moods:
        key = m.lower().replace("-", " ")
        mood_genres.extend(MOOD_TO_GENRES.get(key, []))

    tone = []
    for kw, label in TONE_MOODS.items():
        if _matches(match_text, kw):
            tone.append(label)

    keywords = extract_keywords(q, exclude=reference_title)

    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", match_text)]
    year = years[0] if years else None
    decade = None
    m = re.search(r"\b(19|20)\d0s\b", match_text)
    if m:
        decade = int(re.search(r"\d{4}", m.group()).group())

    runtime_max = None
    runtime_min = None
    m = re.search(r"(?:under|less than|below|shorter than)\s*(\d{2,3})\s*(?:min|mins|minutes|minute|m)\b", match_text)
    if m: runtime_max = int(m.group(1))
    m = re.search(r"(?:over|more than|longer than)\s*(\d{2,3})\s*(?:min|mins|minutes|minute|m)\b", match_text)
    if m: runtime_min = int(m.group(1))
    m = re.search(r"\b(\d{2,3})\s*(?:minute|min)\s*(?:movie|film)?\b", match_text)
    if m and runtime_max is None and runtime_min is None:
        runtime_max = int(m.group(1)) + 10
        runtime_min = max(0, int(m.group(1)) - 10)

    min_rating = None
    m = re.search(r"(?:rated|rating|score(?:d)?|imdb|tmdb)\s*(?:of|above|over|at least)?\s*(\d(?:\.\d)?)", match_text)
    if m: min_rating = float(m.group(1))
    if "highly rated" in match_text or "well rated" in match_text or "good ratings" in match_text:
        min_rating = max(min_rating or 0, 7.0)

    sort = "relevance"
    if any(x in match_text for x in ["new release", "new releases", "latest", "recent", "new movie", "new movies"]): sort = "recent"
    elif any(x in match_text for x in ["trending", "popular", "what's popular", "whats popular"]): sort = "popular"
    elif any(x in match_text for x in ["highest rated", "best rated", "top rated"]): sort = "rating"
    elif any(x in match_text for x in ["classic", "classics", "old movie", "older movie"]): sort = "classic"

    exclusions = []
    m = re.search(r"(?i)\b(?:not|without|no|avoid)\b\s+([\w ,'-]+?)\s*(?:please|thanks|$)", match_text)
    if m:
        raw = m.group(1).strip(", ")
        # stop after a trailing "too X" clause ("not too gory", "not too long")
        raw = re.split(r"\s+too\s+", raw)[0]
        for token in re.split(r"\s*,\s*|\s+and\s+", raw):
            t = token.strip()
            if t in GENRE_ALIASES: exclusions.append(GENRE_ALIASES[t])
            elif t in GENRE_ALIASES.values(): exclusions.append(t)

    themes = []
    for t in _THEME_WORDS:
        if t in match_text:
            themes.append(t)

    confidence = 0.0
    if genres: confidence += 0.30
    if moods: confidence += 0.30
    if reference_title: confidence += 0.25
    if themes: confidence += min(0.10, 0.05 * len(themes))
    if any(x is not None for x in (year, decade, runtime_max, min_rating)): confidence += 0.10
    confidence = round(min(1.0, confidence), 2)

    return {
        "query": q,
        "normalized_query": low,
        "genres": _dedupe(genres),
        "tmdb_genre_ids": [TMDB_GENRE_IDS[g] for g in _dedupe(genres) if g in TMDB_GENRE_IDS],
        "moods": _dedupe(moods),
        "mood_genres": _dedupe(mood_genres),
        "tone": _dedupe(tone),
        "keywords": keywords,
        "reference_title": reference_title,
        "year": year,
        "decade": decade,
        "runtime_min": runtime_min,
        "runtime_max": runtime_max,
        "min_rating": min_rating,
        "sort": sort,
        "exclude_genres": _dedupe(exclusions),
        "themes": themes,
        "confidence": confidence,
    }