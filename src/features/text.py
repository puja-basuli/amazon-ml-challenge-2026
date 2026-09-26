"""Country-independent text normalization and pair similarity features."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


LEGAL = {
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "co": "company", "company": "company",
}
FEATURE_NAMES = [
    "name_ratio", "name_token_ratio", "name_jaro", "name_jaccard",
    "address_ratio", "address_token_ratio", "address_jaccard",
    "number_jaccard", "number_agreement", "number_conflict",
    "name_length_ratio", "address_length_ratio", "name_exact",
    "address_exact", "country_agreement", "candidate_is_s3",
    "name_missing", "address_missing", "shared_postal_like",
    "state_agreement", "postal_agreement", "city_agreement",
    "name_token_overlap", "address_token_overlap",
    "country_conflict", "state_conflict", "postal_conflict", "city_conflict",
]

COMMON_NAME_TOKENS = {
    "and", "the", "company", "co", "corp", "corporation", "inc", "incorporated",
    "limited", "ltd", "private", "pvt", "llc", "llp", "plc", "services", "service",
    "india", "traders", "trading", "enterprises", "enterprise", "group", "business",
}
COMMON_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "near", "opposite", "behind", "floor", "building", "block", "sector", "plot",
    "house", "number", "no", "district", "county", "city", "state", "india", "usa",
}
US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


def normalize(value: str, *, name: bool = False) -> str:
    text = unicodedata.normalize("NFKC", (value or "").casefold()).replace("&", " and ")
    text = text.replace("/", " ")
    text = "".join(char if char.isalnum() else " " for char in text)
    tokens = text.split()
    if name:
        tokens = [LEGAL.get(token, token) for token in tokens]
    else:
        # Only unambiguous street abbreviations are expanded; "st" is left
        # alone because it can mean either "street" or "saint".
        address_aliases = {"rd": "road", "ave": "avenue", "blvd": "boulevard",
                           "dr": "drive", "ln": "lane", "hwy": "highway"}
        tokens = [address_aliases.get(token, token) for token in tokens]
    return " ".join(tokens)


def tokens(text: str) -> set[str]:
    return set(text.split())


def numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?!\w)", text))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def length_ratio(a: str, b: str) -> float:
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0


def normalize_country(value: str) -> str:
    """Normalize known spelling variants while preserving unseen country labels."""
    key = normalize(value)
    aliases = {"us": "us", "u s": "us", "usa": "us", "united states": "us",
               "united states of america": "us", "india": "india", "france": "france"}
    return aliases.get(key, key)


def _location_piece(value: str) -> str:
    return " ".join(normalize(value).split())


def extract_geography(address: str, country: str) -> tuple[str, str, str]:
    """Return conservative (state/region, postal, city/locality) keys.

    The source schema has no separate geography columns, so these are parsed
    from observed comma-delimited address components. Country remains open-set.
    """
    country_key = normalize_country(country)
    parts = [_location_piece(part) for part in (address or "").split(",") if _location_piece(part)]
    numeric_groups = re.findall(r"(?<![A-Za-z0-9])\d{5}(?:-\d{4})?(?![A-Za-z0-9])", address or "")
    postal = ""
    if country_key == "india":
        groups = re.findall(r"(?<!\d)[1-9]\d{5}(?!\d)", address or "")
        postal = groups[-1] if groups else ""
    elif country_key in {"us", "france"}:
        postal = numeric_groups[-1] if numeric_groups else ""
    else:
        groups = re.findall(r"(?<![A-Za-z0-9])[A-Z0-9]{3,10}(?:[ -][A-Z0-9]{2,10})?(?![A-Za-z0-9])", (address or "").upper())
        postal = groups[-1] if groups and any(ch.isdigit() for ch in groups[-1]) else ""
    postal_key = re.sub(r"[^A-Z0-9]", "", postal.upper())
    geo_parts = [part for part in parts if not postal_key or
                 re.sub(r"[^A-Z0-9]", "", part.upper()) != postal_key]

    state = ""
    city = ""
    if country_key == "us":
        state_match = next(((i, code) for i, part in enumerate(geo_parts)
                            for code in re.findall(r"\b[A-Z]{2}\b", part.upper())
                            if code in US_STATE_CODES), None)
        state_idx = state_match[0] if state_match else -1
        if state_idx >= 0:
            state = state_match[1].casefold()
            adjacent = state_idx - 1 if state_idx > 0 else state_idx + 1
            if 0 <= adjacent < len(geo_parts):
                city = geo_parts[adjacent]
        elif geo_parts and len(geo_parts[-1]) <= 2 and geo_parts[-1].upper() in US_STATE_CODES:
            state = geo_parts[-1].casefold()
            city = geo_parts[-2] if len(geo_parts) > 1 else ""
    elif country_key == "india":
        # In the observed Indian records the last comma component is generally
        # the state/UT and the preceding one is a city or locality.
        state = geo_parts[-1] if geo_parts else ""
        city = geo_parts[-2] if len(geo_parts) > 1 else ""
    elif country_key == "france":
        city = geo_parts[-1] if geo_parts else ""
        if postal_key and city == postal.casefold():
            city = geo_parts[-2] if len(geo_parts) > 1 else ""
    elif len(geo_parts) >= 3:
        state, city = geo_parts[-1], geo_parts[-2]
    elif len(geo_parts) == 2:
        city = geo_parts[-1]
    return state, postal_key, city


def informative_tokens(value: str, *, name: bool, limit: int = 2) -> list[str]:
    normalized = normalize(value, name=name)
    stop = COMMON_NAME_TOKENS if name else COMMON_ADDRESS_TOKENS
    candidates = {token for token in normalized.split()
                  if len(token) >= (3 if name else 4) and token not in stop and not token.isdigit()}
    return sorted(candidates, key=lambda token: (-len(token), token))[:limit]


def numeric_components(address: str, limit: int = 2) -> list[str]:
    values = set(re.findall(r"(?<!\w)\d{2,}(?!\w)", normalize(address)))
    return sorted(values, key=lambda value: (-len(value), value))[:limit]


def approximate_grams(value: str, *, name: bool, limit: int = 3) -> list[str]:
    tokens = informative_tokens(value, name=name, limit=4)
    if not tokens:
        return []
    token = tokens[0]
    width = 3 if len(token) >= 3 else 2
    grams = sorted({token[i:i + width] for i in range(len(token) - width + 1)})
    if len(grams) <= limit:
        return grams
    # Spread keys through the token so small typos still leave useful overlap.
    positions = np.linspace(0, len(grams) - 1, num=limit, dtype=int).tolist()
    return [grams[position] for position in dict.fromkeys(positions)]


def keys(name: str, address: str) -> tuple[str, str, str]:
    """Three modest-recall lookup keys; empty keys never form blocks."""
    n = normalize(name, name=True)
    a = normalize(address)
    words = [word for word in n.split() if len(word) >= 3 and word not in COMMON_NAME_TOKENS]
    prefix = "".join(n.split())[:6]
    long_token = sorted(words, key=lambda word: (-len(word), word))[0][:10] if words else ""
    address_words = [word for word in a.split() if len(word) >= 4 and not word.isdigit()
                     and word not in COMMON_ADDRESS_TOKENS]
    address_token = sorted(address_words, key=lambda word: (-len(word), word))[0][:8] if address_words else ""
    numeric = sorted(numbers(a), key=lambda value: (-len(value), value))
    address_key = numeric[0] + ":" + address_token if numeric and address_token else ""
    return prefix, long_token, address_key


def features(left: tuple[str, str, str, str], right: tuple[str, str, str, str]) -> list[float]:
    """Input tuples contain entity_id, name, address, country."""
    name_a, name_b = normalize(left[1], name=True), normalize(right[1], name=True)
    addr_a, addr_b = normalize(left[2]), normalize(right[2])
    geo_a = extract_geography(left[2], left[3])
    geo_b = extract_geography(right[2], right[3])
    nums_a, nums_b = numbers(addr_a), numbers(addr_b)
    post_a = {n for n in nums_a if 5 <= len(n) <= 6}
    post_b = {n for n in nums_b if 5 <= len(n) <= 6}
    return [
        fuzz.ratio(name_a, name_b) / 100,
        fuzz.token_set_ratio(name_a, name_b) / 100,
        JaroWinkler.normalized_similarity(name_a, name_b),
        jaccard(tokens(name_a), tokens(name_b)),
        fuzz.ratio(addr_a, addr_b) / 100,
        fuzz.token_set_ratio(addr_a, addr_b) / 100,
        jaccard(tokens(addr_a), tokens(addr_b)),
        jaccard(nums_a, nums_b),
        float(bool(nums_a & nums_b)),
        float(bool(nums_a and nums_b and not nums_a & nums_b)),
        length_ratio(name_a, name_b),
        length_ratio(addr_a, addr_b),
        float(bool(name_a) and name_a == name_b),
        float(bool(addr_a) and addr_a == addr_b),
        float(bool(left[3]) and normalize_country(left[3]) == normalize_country(right[3])),
        float(right[0].startswith("S3-")),
        float(not name_a or not name_b),
        float(not addr_a or not addr_b),
        float(bool(post_a & post_b)),
        float(bool(geo_a[0]) and geo_a[0] == geo_b[0]),
        float(bool(geo_a[1]) and geo_a[1] == geo_b[1]),
        float(bool(geo_a[2]) and geo_a[2] == geo_b[2]),
        jaccard(tokens(name_a), tokens(name_b)),
        jaccard(tokens(addr_a), tokens(addr_b)),
        float(bool(left[3]) and bool(right[3]) and normalize_country(left[3]) != normalize_country(right[3])),
        float(bool(geo_a[0]) and bool(geo_b[0]) and geo_a[0] != geo_b[0]),
        float(bool(geo_a[1]) and bool(geo_b[1]) and geo_a[1] != geo_b[1]),
        float(bool(geo_a[2]) and bool(geo_b[2]) and geo_a[2] != geo_b[2]),
    ]
