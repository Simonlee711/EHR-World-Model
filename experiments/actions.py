"""
Five-group clinical action space for the EHR world model.

Every medication / infusion code in the MIMIC-IV MEDS extract is mapped to at
most one of five clinically meaningful action groups:

    0  ANTIBIOTIC   systemic antibacterials                      (WHO ATC J01, A07AA, J04AB)
    1  IV_FLUID     crystalloid / colloid volume (no flushes)    (ATC B05AA, B05BB, B05CB/X)
    2  VASOPRESSOR  vasopressors and inotropes                    (ATC C01CA, C01CE, C01CX, H01BA01)
    3  ANALGESIC    opioid and non-opioid analgesics              (ATC N02A, N02BE, N01AH, M01A)
    4  SEDATIVE     sedatives / hypnotics / anaesthetic sedation  (ATC N05BA, N05C, N01AX, N03AA02)

Codes that match none of the groups (e.g. insulin, diuretics, PPIs, laxatives,
anticoagulants, flushes, vaccines) are *not* actions in the 5-action model.

Two code families carry treatment information in the extract:
    MEDICATION//START//<free-text drug name>     (eMAR / pharmacy orders)
    INFUSION_START//<MIMIC itemid>               (ICU inputevents)
Drug infusions are matched on their RxNorm description; volume infusions
(which have no RxNorm mapping) are matched on a fixed list of MIMIC-IV
``d_items`` itemids.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

GROUPS: List[str] = ["ANTIBIOTIC", "IV_FLUID", "VASOPRESSOR", "ANALGESIC", "SEDATIVE"]
GROUP_LABELS: Dict[str, str] = {
    "ANTIBIOTIC":  "Antibiotics",
    "IV_FLUID":    "IV Fluids",
    "VASOPRESSOR": "Vasopressors",
    "ANALGESIC":   "Pain Medications (Analgesics)",
    "SEDATIVE":    "Sedatives",
}
GROUP_ATC: Dict[str, str] = {
    "ANTIBIOTIC":  "J01*, A07AA, J04AB",
    "IV_FLUID":    "B05AA, B05BB, B05CB, B05CX",
    "VASOPRESSOR": "C01CA, C01CE, C01CX, H01BA01",
    "ANALGESIC":   "N02A*, N02BE, N01AH, M01A*",
    "SEDATIVE":    "N05BA, N05C*, N01AX, N03AA02",
}
ACTION_CODES: List[str] = [f"ACTION//{g}" for g in GROUPS]

# ── Drug-name lexicon (case-insensitive, matched on word starts) ────────────────
_LEXICON: Dict[str, List[str]] = {
    "ANTIBIOTIC": [
        "vancomycin", "cefazolin", "cefepime", "ceftriaxone", "ceftazidime", "cefuroxime",
        "cefoxitin", "cefotaxime", "cefotetan", "cefpodoxime", "cefdinir", "cefadroxil",
        "ceftaroline", "ceftolozane", "cephalexin", "piperacillin", "zosyn", "ampicillin",
        "unasyn", "amoxicillin", "augmentin", "penicillin", "nafcillin", "oxacillin",
        "dicloxacillin", "meropenem", "ertapenem", "imipenem", "aztreonam", "ciprofloxacin",
        "levofloxacin", "moxifloxacin", "metronidazole", "flagyl", "azithromycin",
        "erythromycin", "clarithromycin", "clindamycin", "doxycycline", "minocycline",
        "tetracycline", "tigecycline", "linezolid", "daptomycin", "gentamicin", "tobramycin",
        "amikacin", "sulfamethoxazole", "bactrim", "trimethoprim", "nitrofurantoin",
        "macrobid", "rifampin", "rifaximin", "fosfomycin", "colistin", "colistimethate",
        "polymyxin", "dalbavancin", "telavancin", "keflex", "ancef",
    ],
    "VASOPRESSOR": [
        "norepinephrine", "levophed", "phenylephrine", "neo-synephrine", "vasopressin",
        "epinephrine", "adrenalin", "dopamine", "dobutamine", "milrinone", "ephedrine",
        "angiotensin ii", "giapreza", "isoproterenol",
    ],
    "ANALGESIC": [
        "acetaminophen", "ofirmev", "tylenol", "oxycodone", "percocet", "hydromorphone",
        "dilaudid", "morphine", "fentanyl", "hydrocodone", "vicodin", "norco", "tramadol",
        "methadone", "meperidine", "demerol", "codeine", "tapentadol", "remifentanil",
        "sufentanil", "ketorolac", "toradol", "ibuprofen", "naproxen", "celecoxib",
        "diclofenac", "indomethacin", "meloxicam",
    ],
    "SEDATIVE": [
        "propofol", "diprivan", "midazolam", "versed", "lorazepam", "ativan", "diazepam",
        "valium", "dexmedetomidine", "precedex", "ketamine", "alprazolam", "clonazepam",
        "temazepam", "zolpidem", "chlordiazepoxide", "oxazepam", "triazolam",
        "phenobarbital", "pentobarbital", "etomidate",
    ],
    "IV_FLUID": [
        "lactated ringer", "ringers", "plasma-lyte", "plasmalyte", "normosol", "albumin",
        "hetastarch", "hespan", "d5w", "d5 1/2", "d5ns", "d5lr", "dextrose 5%",
        "dextrose 10%", "sodium chloride 0.9%", "sodium chloride 0.45%", "sodium chloride 3%",
        "0.9% sodium chloride", "ns bolus", "nacl 0.9%",
    ],
}
# Group-precedence when a name hits several groups (e.g. "oxycodone-acetaminophen").
_PRIORITY = ["VASOPRESSOR", "SEDATIVE", "ANTIBIOTIC", "ANALGESIC", "IV_FLUID"]

# Never an action: non-systemic routes, flushes, local anaesthetic combos, rescue dextrose,
# antidotes and diagnostics that share a substring with a lexicon entry.
_EXCLUDE = re.compile(
    r"flush|ophth|ointment|cream|otic|eye |eye drop|topical|lotion|gel\b(?!.*oral)|"
    r"lidocaine|bupivacaine|ropivacaine|dextrose 50%|d50|racepinephrine|"
    r"epipen|auto[- ]?injector|naloxone|buprenorphine|suboxone|"
    r"vaginal|enema|irrigation|lock|desmopressin|ddavp|nasal|inhal|neb\b|"
    r"bacitracin|neomycin|guaifenesin|chlorpheniramine|opium|belladonna|"
    r"\bin (?:0\.9%|d5w|ns\b)|bicarbonate",
    re.IGNORECASE,
)

# MIMIC-IV d_items itemids for volume infusions (no RxNorm description in the extract).
IV_FLUID_ITEMIDS: Dict[int, str] = {
    225158: "NaCl 0.9%",
    225828: "Lactated Ringers",
    220949: "Dextrose 5%",
    220950: "Dextrose 10%",
    225159: "NaCl 0.45%",
    225161: "NaCl 3% (hypertonic saline)",
    225823: "D5 1/2NS",
    225825: "D5NS",
    225827: "D5LR",
    225941: "D5 1/4NS",
    220864: "Albumin 5%",
    220862: "Albumin 25%",
    220861: "Albumin (human) 20%",
    220863: "Albumin (human) 4%",
    225174: "Hetastarch (Hespan) 6%",
}


def _compiled(words):
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in words) + r")", re.IGNORECASE)


_PATTERNS = {g: _compiled(ws) for g, ws in _LEXICON.items()}


def classify_name(name: Optional[str]) -> Optional[str]:
    """Map a free-text drug name / description to one of GROUPS (or None)."""
    if not name or not isinstance(name, str) or name.strip().upper() == "UNK":
        return None
    if _EXCLUDE.search(name):
        return None
    hits = [g for g in _PRIORITY if _PATTERNS[g].search(name)]
    return hits[0] if hits else None


def classify_code(code: str, description: Optional[str] = None) -> Optional[str]:
    """Map a MEDS treatment code to an action group.

    MEDICATION//START//<name>  -> classify_name(name)
    INFUSION_START//<itemid>   -> IV_FLUID if itemid in IV_FLUID_ITEMIDS,
                                  else classify_name(RxNorm description)
    """
    if code.startswith("MEDICATION//START//"):
        return classify_name(code[len("MEDICATION//START//"):])
    if code.startswith("INFUSION_START//"):
        try:
            itemid = int(code.split("//")[1])
        except (IndexError, ValueError):
            return None
        if itemid in IV_FLUID_ITEMIDS:
            return "IV_FLUID"
        return classify_name(description)
    return None


def drug_display_name(code: str, description: Optional[str] = None) -> str:
    """Human-readable name for a treatment code."""
    if code.startswith("MEDICATION//START//"):
        return code[len("MEDICATION//START//"):].strip()
    if code.startswith("INFUSION_START//"):
        itemid = int(code.split("//")[1])
        if itemid in IV_FLUID_ITEMIDS:
            return IV_FLUID_ITEMIDS[itemid]
        if description:
            return description.split("\n")[0].strip()
        return f"ICU infusion item {itemid}"
    return code
