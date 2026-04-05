# ============================================
# backend/onet_india_filter.py
# ============================================

CATCHALL_SUFFIXES = [
    ", all other",
    ", all others",
    ", n.e.c.",
    ", n.e.c",
]

ILLEGAL_OR_NONEXISTENT = [
    "cannabis",
    "marijuana",
    "gambling",
    "casino",
    "bartender",
    "barista",
    "sommelier",
]

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

US_SPECIFIC_ROLES = [
    "bail bond",
    "physician assistant",
    "nurse practitioner",
    "nurse anesthetist",
    "coroner",
    "title examiner",
    "abstractors",
    "postal service",
    "transportation security screener",
    "probation officer",
    "correctional officer",
    "bailiff",
    "administrative law judge",
    "legislators",
]

EXCLUDE_SECTORS = {"Military"}


def _is_catchall(title_lower: str) -> bool:
    for suffix in CATCHALL_SUFFIXES:
        if title_lower.endswith(suffix):
            return True
    return False


def _is_illegal_or_nonexistent(title_lower: str) -> bool:
    return any(term in title_lower for term in ILLEGAL_OR_NONEXISTENT)


def _is_low_education(title_lower: str) -> bool:
    return any(term in title_lower for term in LOW_EDUCATION_ROLES)


def _is_us_specific(title_lower: str) -> bool:
    return any(term in title_lower for term in US_SPECIFIC_ROLES)


def is_india_relevant(onet_code: str, title: str, sector: str) -> bool:
    if sector in EXCLUDE_SECTORS:
        return False

    title_lower = title.lower().strip()

    if _is_catchall(title_lower):
        return False

    if _is_illegal_or_nonexistent(title_lower):
        return False

    if _is_low_education(title_lower):
        return False

    if _is_us_specific(title_lower):
        return False

    return True
