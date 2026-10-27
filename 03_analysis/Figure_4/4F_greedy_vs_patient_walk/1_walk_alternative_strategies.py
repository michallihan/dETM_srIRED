import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_HIGH_CONF_KCAT_CSV  # noqa: E402
from variant_rounds import ROUNDS, variants_per_round  # noqa: E402

# The variant every branch starts from: the campaign's round-1 winner.
ANCHOR = "T241A"

# Where the branches part company, and how many are followed.
BRANCH_ROUND = 2
N_BRANCHES = 4

# A candidate carrying more mutations than this besides the anchor is passed over: one round of
# mutagenesis is not expected to deliver them all at once.
MAX_EXTRA_MUTATIONS = 2

# What the campaign fixed after each round, as recorded, indexed by round. Round 0 is the
# starting point and round 2 repeats round 1: the round-2 library was still a T241A library.
CAMPAIGN_VARIANTS_BY_ROUND = [
    "WT",
    "T241A",
    "T241A",
    "S68P-T241A",
    "S68P-T241A-H244Y",
    "S68P-P203Q-T241A-H244Y",
]

Y_MAX = 140

CAMPAIGN_COLOUR = "#E8384F"
BRANCH_COLOURS = ("#D9901F", "#3B7DD8", "#4BA36B", "#8E5FBF", "#C2578A", "#5BA8A0")

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def enrichment(fitness_score):
    """The fitness score on the enrichment scale the evaluation reports."""
    return float(np.exp(fitness_score))


def mutations(mut_id):
    """The substitutions a variant carries, as a set."""
    return set() if mut_id == "WT" else set(mut_id.split("-"))


def load_scored_rounds():
    """Scores with the modelled apparent kcat, and the scored variants in each round."""
    columns = ["mut_id", "fitness_final", "sigma_final", "kcat_app", "kcat_app_error"]
    fitness = pd.read_csv(FITNESS_HIGH_CONF_KCAT_CSV)[columns]
    fitness = fitness.drop_duplicates("mut_id").set_index("mut_id")

    present = variants_per_round()
    present = present[present["mut_id"].isin(fitness.index)]
    libraries = {r: sorted(set(table["mut_id"])) for r, table in present.groupby("round")}
    return fitness, libraries


class Walks:
    """Following one lineage from round to round through the libraries that were built."""

    def __init__(self, fitness, libraries):
        self.score = fitness["fitness_final"].to_dict()
        self.kcat = fitness["kcat_app"].to_dict()
        self.kcat_error = fitness["kcat_app_error"].to_dict()
        self.libraries = libraries
        self.mutations = {m: mutations(m) for m in self.score}
        self.peak = max(self.score, key=self.score.get)

    def carriers(self, mutation, round_number):
        """Scored variants in a round's library carrying a given mutation."""
        return [m for m in self.libraries[round_number] if mutation in self.mutations[m]]

    def founder(self, start):
        """The candidate's own mutation that still has carriers in every later round.

        Where more than one qualifies the one with the most carriers downstream is taken;
        the count of qualifying mutations is returned so an ambiguous choice can be flagged.
        """
        later = [r for r in ROUNDS if r > BRANCH_ROUND]
        surviving = [mutation for mutation in sorted(self.mutations[start] - {ANCHOR})
                     if all(self.carriers(mutation, r) for r in later)]
        if not surviving:
            return None, 0
        surviving.sort(key=lambda m: -sum(len(self.carriers(m, r)) for r in later))
        return surviving[0], len(surviving)

    def walk(self, start, founder):
        """Each round's best carrier of the founder mutation, from that round's library."""
        path = [start]
        for round_number in ROUNDS:
            if round_number > BRANCH_ROUND:
                path.append(max(self.carriers(founder, round_number), key=self.score.get))
        return path


def choose_branches(walks):
    """Walk the round-2 ranking, recording why each candidate is taken or passed over."""
    ranking = sorted(walks.libraries[BRANCH_ROUND], key=walks.score.get, reverse=True)
    kept, log = [], []

    for rank, candidate in enumerate(ranking, start=1):
        if len(kept) == N_BRANCHES:
            log.append((rank, candidate, "not reached", None))
            continue
        if candidate == ANCHOR or ANCHOR not in walks.mutations[candidate]:
            reason = "does not carry the anchor" if candidate != ANCHOR else "is the anchor"
            log.append((rank, candidate, reason, None))
            continue
        if len(walks.mutations[candidate] - {ANCHOR}) > MAX_EXTRA_MUTATIONS:
            log.append((rank, candidate, f"more than {MAX_EXTRA_MUTATIONS} mutations", None))
            continue

        founder, n_surviving = walks.founder(candidate)
        if founder is None:
            log.append((rank, candidate, "dies out", None))
            continue

        path = walks.walk(candidate, founder)
        if path[-1] == walks.peak:
            log.append((rank, candidate, "same lineage as the highest-scoring variant", founder))
            continue
        taken = next((branch for branch in kept if branch["founder"] == founder), None)
        if taken:
            log.append((rank, candidate, f"same lineage as #{taken['rank']}", founder))
            continue

        kept.append(dict(rank=rank, start=candidate, founder=founder,
                         ambiguous_founder=n_surviving > 1))
        log.append((rank, candidate, "kept", founder))

    return kept, log


def changes(variant, previous):
    """Which mutations a round's pick gained and lost against the one before it."""
    if previous is None:
        return "", ""
    gained = sorted(mutations(variant) - mutations(previous), key=lambda m: int(m[1:-1]))
    lost = sorted(mutations(previous) - mutations(variant), key=lambda m: int(m[1:-1]))
    return " ".join(gained), " ".join(lost)


def build_trajectories(walks, sigma, branches):
    """Every branch, plus the campaign, as one long table."""
    rows = []

    def add(label, founder, ambiguous, path):
        for round_number, variant in enumerate(path):
            previous = path[round_number - 1] if round_number else None
            gained, lost = changes(variant, previous)
            score = walks.score[variant]
            spread = sigma.get(variant, np.nan)
            rows.append(dict(
                branch=label, founder=founder, round=round_number, mut_id=variant,
                fitness_final=score, sigma_final=float(spread),
                enrichment=enrichment(score),
                enrichment_low=enrichment(score - spread),
                enrichment_high=enrichment(score + spread),
                kcat_app=walks.kcat[variant],
                kcat_app_error=walks.kcat_error[variant],
                gained=gained, lost=lost,
                ambiguous_founder=ambiguous,
            ))

    add("campaign", "", False, CAMPAIGN_VARIANTS_BY_ROUND)
    for branch in branches:
        add(f"#{branch['rank']}", branch["founder"], branch["ambiguous_founder"],
            ["WT", ANCHOR] + walks.walk(branch["start"], branch["founder"]))
    return pd.DataFrame(rows)


def plot(trajectories, branches):
    """Apparent kcat against round: the campaign, and the branches a screen could take."""
    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    for branch, colour in zip(branches, BRANCH_COLOURS):
        draw(ax, trajectories, f"#{branch['rank']}", colour, 2.0, 5.5)
    draw(ax, trajectories, "campaign", CAMPAIGN_COLOUR, 2.4, 6.0)

    ax.set_xlabel("Round of evolution")
    ax.set_ylabel("$k_{cat}$(app)  (s$^{-1}$)")
    ax.set_xticks(range(len(ROUNDS) + 1))
    ax.set_xlim(-0.25, len(ROUNDS) + 0.3)
    ax.set_ylim(0, Y_MAX)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)

    handles = [plt.Line2D([], [], color=CAMPAIGN_COLOUR, lw=2.4, marker="o", ms=6,
                          label="Campaign")]
    handles += [plt.Line2D([], [], color=colour, lw=2, marker="o", ms=5.5,
                           label=f"#{branch['rank']}  {branch['start']}")
                for branch, colour in zip(branches, BRANCH_COLOURS)]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7.2,
              handlelength=1.5, borderpad=0, labelspacing=0.35)

    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(THIS_DIR / f"trajectories.{extension}", dpi=400)
    plt.close(fig)


def plot_campaign_and_best_hit(trajectories, branches):
    """The campaign against the round-2 best hit alone."""
    best_hit = next((branch for branch in branches if branch["rank"] == 1), None)
    if best_hit is None:
        print("\nno branch was kept at rank 1; campaign_and_best_hit not drawn")
        return

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    draw(ax, trajectories, f"#{best_hit['rank']}", BRANCH_COLOURS[0], 2.2, 6.0)
    draw(ax, trajectories, "campaign", CAMPAIGN_COLOUR, 2.4, 6.0)

    ax.set_xlabel("Round of evolution")
    ax.set_ylabel("$k_{cat}$(app)  (s$^{-1}$)")
    ax.set_xticks(range(len(ROUNDS) + 1))
    ax.set_xlim(-0.25, len(ROUNDS) + 0.3)
    ax.set_ylim(0, Y_MAX)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)

    handles = [plt.Line2D([], [], color=CAMPAIGN_COLOUR, lw=2.4, marker="o", ms=6,
                          label="This campaign"),
               plt.Line2D([], [], color=BRANCH_COLOURS[0], lw=2.2, marker="o", ms=6,
                          label="Best hit in round 2")]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8,
              handlelength=1.5, borderpad=0, labelspacing=0.35)

    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(THIS_DIR / f"campaign_and_best_hit.{extension}", dpi=400)
    plt.close(fig)


def draw(ax, trajectories, branch, colour, width, marker_size):
    """One walk in apparent kcat, with the error the conversion carries through."""
    walk = trajectories[trajectories["branch"] == branch]
    x = walk["round"].to_numpy()
    y = walk["kcat_app"].to_numpy()

    ax.errorbar(x, y, yerr=walk["kcat_app_error"].to_numpy(), fmt="none", ecolor=colour,
                elinewidth=1, capsize=2, alpha=0.45, zorder=2)
    ax.plot(x, y, color=colour, lw=width, zorder=3, solid_capstyle="round")
    ax.plot(x, y, "o", ms=marker_size, color=colour, mec="white", mew=1.1, zorder=4)


def main():
    fitness, libraries = load_scored_rounds()
    sigma = fitness["sigma_final"].to_dict()
    for round_number in ROUNDS:
        print(f"round {round_number}: {len(libraries[round_number])} scored variants")

    walks = Walks(fitness, libraries)
    print(f"\nhighest-scoring variant in the dataset: {walks.peak} "
          f"(kcat_app {walks.kcat[walks.peak]:.1f} s-1)")

    branches, log = choose_branches(walks)
    evaluated = [entry for entry in log if entry[2] != "not reached"]
    selection = pd.DataFrame(
        [dict(rank=rank, mut_id=candidate, fitness_final=walks.score[candidate],
              enrichment=enrichment(walks.score[candidate]),
              founder=founder or "", outcome=reason)
         for rank, candidate, reason, founder in evaluated])
    selection.to_csv(THIS_DIR / "branch_selection.csv", index=False)

    print(f"\nround-{BRANCH_ROUND} ranking, down to branch {N_BRANCHES}:")
    for rank, candidate, reason, _ in evaluated:
        print(f"  {rank:3d}  {candidate:32s} {walks.kcat[candidate]:6.1f}  {reason}")

    trajectories = build_trajectories(walks, sigma, branches)
    trajectories.to_csv(THIS_DIR / "trajectories.csv", index=False)

    print("\nwalks (kcat_app, s-1):")
    for label in ["campaign"] + [f"#{branch['rank']}" for branch in branches]:
        path = trajectories[trajectories["branch"] == label]
        flag = ("   [founder ambiguous: chosen on carrier count]"
                if bool(path["ambiguous_founder"].iloc[0]) else "")
        steps = " -> ".join(f"{m}({k:.1f})" for m, k in zip(path["mut_id"], path["kcat_app"]))
        print(f"  {label:9s} {steps}{flag}")

    plot(trajectories, branches)
    plot_campaign_and_best_hit(trajectories, branches)
    print("\nwrote trajectories.csv, branch_selection.csv, trajectories.svg / .png "
          "and campaign_and_best_hit.svg / .png")


if __name__ == "__main__":
    raise SystemExit(main())
