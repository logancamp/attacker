"""
PHASE 2: FEATURE EXTRACTION
Extract 16 features from each query in the input dataset.

Input CSV expected columns:
    AnonID      - user identifier
    Query       - query string
    QueryTime   - timestamp (e.g. 2006-03-01 16:01:20)
    ItemRank    - rank of clicked result (optional, can be empty)
    ClickURL    - URL clicked (optional, can be empty)
    Label       - 'real' or 'fake'
    Role        - 'train' (held-out users) or 'target' (user being attacked)

Output: output_dir/query_features.pkl
    List of dicts, one per query, each containing all 16 features
    plus metadata (AnonID, Query, QueryTime, Label, Role, query_id)
"""

import argparse
import os
import pickle
from collections import Counter

import geonamescache
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Feature Extraction")
    parser.add_argument("--input",      required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="output", help="Directory for output files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_path):
    """Load CSV and validate required columns exist."""
    df = pd.read_csv(input_path)

    required = ["AnonID", "Query", "QueryTime", "Label", "Role"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Normalise types
    df["QueryTime"] = pd.to_datetime(df["QueryTime"])
    df["ClickURL"]  = df.get("ClickURL",  pd.Series("", index=df.index)).fillna("")
    df["Query"]     = df["Query"].fillna("").astype(str)
    df["Label"]     = df["Label"].str.strip().str.lower()
    df["Role"]      = df["Role"].str.strip().str.lower()

    print(f"  Loaded {len(df):,} rows | "
          f"{df['AnonID'].nunique()} users | "
          f"{(df['Label']=='real').sum()} real | "
          f"{(df['Label']=='fake').sum()} fake")
    return df

# ---------------------------------------------------------------------------
# Helper: term popularity across whole corpus
# ---------------------------------------------------------------------------

def compute_term_popularity(df):
    """
    STEP 14 helper: frequency of each term across the entire query corpus.
    Returns dict {term: relative_frequency}.
    Used later as a per-query feature (sum of term frequencies in corpus).
    """
    counter = Counter()
    for q in df["Query"]:
        counter.update(str(q).lower().split())
    total = sum(counter.values()) or 1
    return {t: c / total for t, c in counter.items()}


# ---------------------------------------------------------------------------
# Helper: city / country name sets
# ---------------------------------------------------------------------------

def get_location_sets():
    """
    STEP 13 helper: build sets of lower-cased city and country names
    using the geonamescache package (no external API needed).
    """
    gc = geonamescache.GeonamesCache()
    cities    = {c["name"].lower() for c in gc.get_cities().values()}
    countries = {c["name"].lower() for c in gc.get_countries().values()}
    return cities, countries


# ---------------------------------------------------------------------------
# Helper: click count per (user, timestamp)
# ---------------------------------------------------------------------------

def build_click_counts(df):
    """
    STEP 9 helper: AOL rows with a non-empty ClickURL represent a click event.
    Group by (AnonID, QueryTime) to get clicks per query event.
    Returns a dict keyed by (AnonID, QueryTime) -> int.
    """
    clicks = (
        df[df["ClickURL"] != ""]
        .groupby(["AnonID", "QueryTime"])
        .size()
        .to_dict()
    )
    return clicks


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features(df):
    """
    Extract all 16 features for every query row.
    Returns a list of feature dicts (one per row).
    """

    # --- initialise shared resources ---

    # STEP 15: SBERT model for contextual embeddings
    #          replaces ODP topic features from the original paper
    print("  Loading SBERT model (downloads ~90 MB on first run)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # STEP 12: English spell checker
    print("  Loading spell checker...")
    spell = SpellChecker()

    # STEP 13: Location name sets
    print("  Loading city / country sets...")
    cities, countries = get_location_sets()

    # STEP 14: Term popularity across corpus
    print("  Computing corpus-level term popularity...")
    term_popularity = compute_term_popularity(df)

    # STEP 9: Pre-compute click counts
    click_counts = build_click_counts(df)

    # STEP 15: Batch-encode all queries at once (much faster than one-by-one)
    print("  Batch encoding queries with SBERT...")
    all_queries  = df["Query"].tolist()
    embeddings   = sbert.encode(all_queries, batch_size=64, show_progress_bar=True)

    # --- per-query extraction ---

    features_list = []

    for row_pos, (idx, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="  Extracting features")
    ):
        query  = str(row["Query"]).lower()
        terms  = query.split()
        termset = set(terms)

        # STEP 7: Raw Unix timestamp
        timestamp = row["QueryTime"].timestamp()

        # STEP 8: Day of week (0=Mon … 6=Sun) and hour of day (0-23)
        day_of_week = row["QueryTime"].dayofweek
        hour_of_day = row["QueryTime"].hour
        is_weekend  = int(day_of_week >= 5)

        # STEP 9: Number of clicked results for this query event
        num_clicks = click_counts.get((row["AnonID"], row["QueryTime"]), 0)

        # STEP 10: Average term frequency within this query
        #          (ratio of token count to unique token count)
        term_freq = len(terms) / max(len(termset), 1)

        # STEP 11: Word count and character count
        num_terms = len(terms)
        num_chars = len(query)

        # STEP 12: Spelling errors
        misspelled         = spell.unknown(terms)
        num_spelling_errors = len(misspelled)
        has_spelling_error  = int(num_spelling_errors > 0)

        # STEP 13: Location terms (city / country mentions)
        query_cities    = [t for t in terms if t in cities]
        query_countries = [t for t in terms if t in countries]
        has_city        = int(bool(query_cities))
        has_country     = int(bool(query_countries))
        has_location    = int(has_city or has_country)
        city_name       = query_cities[0]    if query_cities    else ""
        country_name    = query_countries[0] if query_countries else ""

        # STEP 14: Query term popularity
        #          (sum of each term's corpus frequency — popular terms = high weight)
        term_weight = sum(term_popularity.get(t, 0.0) for t in terms)

        # STEP 15: SBERT contextual embedding (384-d vector)
        #          Replaces ODP S_Level2Cat and D_TreeDistance features
        embedding = embeddings[row_pos]

        features_list.append({
            # --- metadata (not features, used for bookkeeping) ---
            "query_id":   idx,           # original dataframe index
            "AnonID":     row["AnonID"],
            "Query":      row["Query"],  # original casing preserved
            "QueryTime":  row["QueryTime"],
            "Label":      row["Label"],  # 'real' / 'fake'
            "Role":       row["Role"],   # 'train' / 'target'

            # --- STEP 7 ---
            "timestamp":            timestamp,

            # --- STEP 8 ---
            "day_of_week":          day_of_week,
            "hour_of_day":          hour_of_day,
            "is_weekend":           is_weekend,

            # --- STEP 9 ---
            "num_clicks":           num_clicks,

            # --- STEP 10 ---
            "term_freq":            term_freq,

            # --- STEP 11 ---
            "num_terms":            num_terms,
            "num_chars":            num_chars,

            # --- STEP 12 ---
            "num_spelling_errors":  num_spelling_errors,
            "has_spelling_error":   has_spelling_error,

            # --- STEP 13 ---
            "has_city":             has_city,
            "has_country":          has_country,
            "has_location":         has_location,
            "city_name":            city_name,
            "country_name":         country_name,

            # --- STEP 14 ---
            "term_weight":          term_weight,

            # --- STEP 15 ---
            "embedding":            embedding,   # np.ndarray shape (384,)
        })

    return features_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[PHASE 2] Feature Extraction")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output_dir}/query_features.pkl\n")

    df = load_data(args.input)
    features = extract_features(df)

    output_path = os.path.join(args.output_dir, "query_features.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"\n  Saved {len(features):,} query feature records -> {output_path}")


if __name__ == "__main__":
    main()
