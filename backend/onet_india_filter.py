# ============================================
# onet_india_filter.py
# Filter O*NET occupations for India relevance
#
# Drop in to onet_loader.py — call filter_for_india()
# on the occupations list before fetching RIASEC/skills.
#
# Three-layer filtering:
#   1. Catch-all titles ("All Other", "N.E.C.")
#   2. Illegal / non-existent in India
#   3. Low-education trades (job zone 1–2 equivalent)
#   4. US-specific regulatory / structural roles
#   5. Military sector (SOC prefix 55)
# ============================================


# ── Layer 1: Catch-all / placeholder titles ────────────────────────────────
# These are O*NET bucket codes with no specific career identity.
# A student cannot pursue "Engineers, All Other" — it maps to nothing actionable.
CATCHALL_SUFFIXES = [
    ", all other",
    ", all others",
    ", n.e.c.",
    ", n.e.c",
]

# ── Layer 2: Illegal or structurally non-existent in India ────────────────
# Careers either prohibited by Indian law or where the role simply
# does not exist as a formal profession in India.
ILLEGAL_OR_NONEXISTENT = [
    "cannabis",
    "marijuana",
    "gambling",          # covers: gambling dealer, gambling manager, gambling cage, etc.
    "casino",
    "bartender",         # alcohol service roles not a career track in India
    "barista",           # not a formal career path in India
    "sommelier",
]

# ── Layer 3: Low-education / no post-12th study required ──────────────────
# These are trade/craft jobs that don't require further education.
# Students using a career guidance app are asking "what to study next",
# not "what trade can I learn without studying".
LOW_EDUCATION_ROLES = [
    "laborer",
    "laborers",
    "cleaner",
    "cleaners",
    "maid",
    "maids",
    "janitor",
    "janitorial",
    "dishwasher",
    "packer",
    "packager",
    "cashier",
    "parking attendant",
    "baggage porter",
    "bellhop",
    "waiter",
    "waitress",
    "waitresses",
    "food server",
    "fast food",
    "counter worker",
    "shampooer",
    "locker room attendant",
    "usher",
    "ticket taker",
    "crossing guard",
    "school bus monitor",
    "nanny",
    "childcare worker",
    "costume attendant",
    "motion picture projectionist",
    "taxi driver",
    "shuttle driver",
    "light truck driver",
    "driver/sales worker",
    "pest control",
    "tree trimmer",
    "grounds maintenance",
    "landscaping and groundskeeping worker",
    "farmworker",
    "logging worker",
    "faller",
    "log grader",
    "fishing and hunting",
    "door-to-door sales",
    "street vendor",
    "telemarketers",
    "models",
    "concierge",
    "tour guide",
    "travel guide",
    "amusement and recreation attendant",
    "recreation worker",
    "residential advisor",
]

# ── Layer 4: US-specific structural / regulatory roles ────────────────────
# Roles that exist because of US-specific laws, institutions, or systems
# that have no meaningful Indian equivalent a student could pursue.
US_SPECIFIC_ROLES = [
    "bail bond",
    "physician assistant",       # PA credential doesn't exist in India
    "nurse practitioner",        # NP prescribing rights don't exist in India
    "nurse anesthetist",         # CRNA model doesn't exist
    "coroner",                   # India uses forensic medicine officers under magistrates
    "title examiner",            # US property law specific
    "abstractors",               # US property law specific
    "postal service",            # India Post is a govt department, not open career path post-12th
    "transportation security screener",   # TSA is US federal agency
    "probation officer",         # US criminal justice system specific
    "correctional officer",      # Prison system structure differs
    "bailiff",                   # Doesn't exist in Indian courts
    "administrative law judge",  # US system specific
    "legislators",               # Elected position, not a career path via education
]

# ── Layer 5: Entire sectors to exclude ────────────────────────────────────
# Military sector (SOC prefix 55) — US Armed Forces structure,
# not applicable to Indian defence career pathways.
EXCLUDE_SECTORS = {"Military"}


def _is_catchall(title_lower: str) -> bool:
    """Returns True if title is a catch-all placeholder."""
    for suffix in CATCHALL_SUFFIXES:
        if title_lower.endswith(suffix):
            return True
    return False


def _is_illegal_or_nonexistent(title_lower: str) -> bool:
    """Returns True if career is illegal or non-existent in India."""
    return any(term in title_lower for term in ILLEGAL_OR_NONEXISTENT)


def _is_low_education(title_lower: str) -> bool:
    """Returns True if career doesn't require post-12th study."""
    return any(term in title_lower for term in LOW_EDUCATION_ROLES)


def _is_us_specific(title_lower: str) -> bool:
    """Returns True if career is specific to US legal/structural system."""
    return any(term in title_lower for term in US_SPECIFIC_ROLES)


def is_india_relevant(onet_code: str, title: str, sector: str) -> bool:
    """
    Master filter — returns True if the occupation should be included
    in recommendations for Indian Class 12 students.

    Args:
        onet_code  — e.g. "17-2051.00"
        title      — e.g. "Civil Engineers"
        sector     — mapped sector string e.g. "Engineering"

    Returns:
        True  → include in recommendations
        False → exclude
    """
    # Layer 5: sector-level block
    if sector in EXCLUDE_SECTORS:
        return False

    title_lower = title.lower().strip()

    # Layer 1: catch-all titles
    if _is_catchall(title_lower):
        return False

    # Layer 2: illegal / non-existent in India
    if _is_illegal_or_nonexistent(title_lower):
        return False

    # Layer 3: low-education trades
    if _is_low_education(title_lower):
        return False

    # Layer 4: US-specific structural roles
    if _is_us_specific(title_lower):
        return False

    return True


# ── Integration: drop into onet_loader.py ─────────────────────────────────
#
# In fetch_all_onet_careers(), after calling _fetch_all_occupations():
#
#   occupations = _fetch_all_occupations(api_key)
#
#   # ── Apply India filter ──────────────────────────────────
#   from onet_india_filter import is_india_relevant
#   occupations = [
#       occ for occ in occupations
#       if is_india_relevant(
#           occ["code"],
#           occ["title"],
#           SOC_TO_SECTOR.get(_get_soc_prefix(occ["code"]), "General")
#       )
#   ]
#   print(f"After India filter: {len(occupations)} occupations")
#   # ────────────────────────────────────────────────────────
#
#   for i, occ in enumerate(occupations):
#       ...  # rest of the loop unchanged
