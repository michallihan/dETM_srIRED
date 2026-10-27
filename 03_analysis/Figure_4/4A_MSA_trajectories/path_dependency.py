import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import COLLAPSED_UNIQUE_VARIANTS_CSV  # noqa: E402

WILD_TYPE = (
    "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPL"
    "GDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLG"
    "ADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVA"
    "TVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPS*"
)

BRANCHES = ("dETM", "MSA")

# Sequence column of each source table, and where the sequences come from.
DETM_SOURCE_CSV = COLLAPSED_UNIQUE_VARIANTS_CSV
DETM_SEQUENCE_COLUMN = 3          # aa_sequence
MSA_SOURCE_FASTA = THIS_DIR / "blast_top5000.fasta"

# The alignment is scored only at the positions the campaign visited, plus the
# neighbouring positions of the same structural region.
MSA_CALL_POSITIONS = (37, 38, 68, 69, 72, 203, 241, 244)

# Nested backgrounds along each trajectory: (mutation, background it is nested in).
# A background of None means the mutation is filtered out of the full set.
BACKGROUND_CHAINS = {
    "dETM": (("T241A", None), ("S68P", "T241A"), ("P203Q", "S68P")),
    "MSA": (("T241A", None), ("S68C", "T241A"), ("S68P", "T241A"),
            ("P203T", "S68C"), ("P203G", "S68P")),
}

# Amino-acid frequency panels: (position, background, residues).
# Residues are the wild-type residue followed by the two mutations of interest;
# everything else at that position is pooled as "Other".
FREQUENCY_PANELS = {
    "dETM": ((241, None, "TAV"),
             (68, None, "SPV"),
             (68, "T241A", "SPC"),
             (203, "S68P", "PQG"),
             (244, "P203Q", "HYA")),
    "MSA": ((241, None, "TAV"),
            (68, None, "SPV"),
            (68, "T241A", "SPC"),
            (203, "S68C", "PQT"),
            (203, "S68P", "PQG"),
            (244, "P203T", "HYD"),
            (244, "P203G", "HYG")),
}

# How the wild-type bar of a frequency panel is obtained. See the module docstring.
WILD_TYPE_SHARE = {"dETM": "remainder", "MSA": "counted"}

# Sets the per-position mutation occurrence plot is drawn for. The campaign is shown as
# a whole; the alignment is shown for every background as well.
OCCURRENCE_SETS = {
    "dETM": (None,),
    "MSA": (None, "T241A", "S68C", "S68P", "P203T", "P203G"),
}

FREQUENCY_COLOURS = ("turquoise", "salmon", "gold", "gray")

AMINO_ACID_COLOURS = {
    "A": "lawngreen", "C": "royalblue", "D": "gainsboro", "E": "mediumpurple",
    "F": "orangered", "G": "gold", "H": "aquamarine", "I": "hotpink",
    "K": "lightseagreen", "L": "olive", "M": "navy", "N": "maroon",
    "P": "lightskyblue", "Q": "darkviolet", "R": "coral", "S": "gray",
    "T": "pink", "V": "forestgreen", "W": "crimson", "Y": "goldenrod",
}

PLOT_FONT = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

MUTATION_PATTERN = re.compile(r"^[A-Z](\d+)([A-Z])$")
NON_MUTATION_VALUES = ("frameshift", "no_translation")


def branch_dir(branch):
    """Folder holding one branch's tables and figures."""
    return THIS_DIR / branch


def called_mutations_csv(branch, background=None):
    """Mutation-called table for a branch, either the full set or one background."""
    stem = "called_mutations" if background is None else f"called_mutations_in_{background}"
    return branch_dir(branch) / f"{stem}.csv"


def occurrence_stem(background=None):
    """Filename stem of the per-position occurrence outputs."""
    return "mutation_occurrence_all" if background is None else f"mutation_occurrence_in_{background}"


def call_all_differences(sequence):
    """Every position at which `sequence` differs from the wild type (dETM rule)."""
    mutations = []
    for i in range(min(len(WILD_TYPE), len(sequence))):
        if WILD_TYPE[i] != sequence[i]:
            mutations.append(f"{WILD_TYPE[i]}{i + 1}{sequence[i]}")
    if len(sequence) != len(WILD_TYPE):
        mutations.append(f"LengthMismatch_{len(sequence)}vs{len(WILD_TYPE)}")
    return mutations


def call_at_positions(sequence, positions=MSA_CALL_POSITIONS):
    """A call at each tracked position, identity calls included (MSA rule).

    Positions beyond the end of either sequence are left out.
    """
    calls = []
    for position in positions:
        i = position - 1
        if i < len(WILD_TYPE) and i < len(sequence):
            calls.append(f"{WILD_TYPE[i]}{position}{sequence[i]}")
    return calls


def residue_counts(mutation_strings, position):
    """Count the residues called at `position` across a column of mutation strings."""
    counts = {}
    for mutations in mutation_strings:
        if not isinstance(mutations, str) or mutations in NON_MUTATION_VALUES:
            continue
        for call in mutations.split("-"):
            match = MUTATION_PATTERN.match(call)
            if match and int(match.group(1)) == position:
                residue = match.group(2)
                counts[residue] = counts.get(residue, 0) + 1
    return counts


def parse_mutation_string(mutation_string):
    """Split a mutation string into (from, position, to) triples, dropping non-calls."""
    if not mutation_string or mutation_string.upper() in ("WT", "WILDTYPE"):
        return []
    parsed = []
    for call in mutation_string.split("-"):
        if len(call) < 3:
            continue
        position = call[1:-1]
        if position.isdigit():
            parsed.append((call[0], int(position), call[-1]))
    return parsed
