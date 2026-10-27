import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import BIOPHYSICAL_DATA_CSV, FITNESS_HIGH_CONF_CSV  # noqa: E402

SUBSTITUTION = "S68P"
KCAT, KCAT_ERROR = "kcat", "kcat (error) [s-1]"
KM, KM_ERROR = "KM", "KM (error) [mM]"

WITH_COLOUR = "black"
WITHOUT_COLOUR = "0.55"

# Arial, or the closest thing installed. Liberation Sans and Helvetica carry Arial's metrics,
# so a figure drawn with either lays out identically; run this on a machine with Arial itself
# if the embedded font matters.
FONT_CANDIDATES = ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans")


def available_font():
    """The first of the candidates matplotlib can actually find."""
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in FONT_CANDIDATES:
        if candidate in installed:
            return candidate
    return FONT_CANDIDATES[-1]


def style():
    """One sans-serif face throughout, including the maths in the axis labels."""
    face = available_font()
    family = [face, *FONT_CANDIDATES]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(family),
        "font.size": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": face,
        "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold",
        "mathtext.default": "it",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def normalise(identifiers):
    """One spelling of a mutation identifier, so the two tables join. As in S7."""
    return identifiers.astype(str).str.replace(",", "-", regex=False).str.strip()


def load():
    """S7's variants -- kinetics and a high-confidence fitness score -- labelled by S68P."""
    biophysics = pd.read_csv(BIOPHYSICAL_DATA_CSV)
    fitness = pd.read_csv(FITNESS_HIGH_CONF_CSV)

    biophysics["variant"] = normalise(biophysics["aa_variant"])
    fitness["variant"] = normalise(fitness["mut_id"])

    table = biophysics.merge(fitness[["variant", "fitness_final"]], on="variant", how="left")
    table = table.dropna(subset=[KCAT, KM, "fitness_final"]).copy()
    table["carries_S68P"] = table["aa_variant"].str.contains(SUBSTITUTION, regex=False)

    columns = ["aa_variant", "nham", "carries_S68P", "fitness_final",
               KCAT, KCAT_ERROR, KM, KM_ERROR]
    return table[columns].sort_values(["carries_S68P", KM]).reset_index(drop=True)


def ancova(table):
    """Fit kcat ~ KM * S68P by least squares and test the interaction term."""
    km = table[KM].to_numpy(float)
    group = table["carries_S68P"].to_numpy(float)
    kcat = table[KCAT].to_numpy(float)

    design = np.column_stack([np.ones_like(km), km, group, km * group])
    names = ["intercept", "KM", "S68P", "KM x S68P"]

    coefficients, *_ = np.linalg.lstsq(design, kcat, rcond=None)
    observations, parameters = design.shape
    residuals = kcat - design @ coefficients
    variance = residuals @ residuals / (observations - parameters)
    errors = np.sqrt(np.diag(variance * np.linalg.pinv(design.T @ design)))
    t_statistics = coefficients / errors
    p_values = 2 * stats.t.sf(np.abs(t_statistics), observations - parameters)

    total = kcat - kcat.mean()
    r_squared = 1 - residuals @ residuals / (total @ total)
    return {
        "names": names, "coefficients": coefficients, "errors": errors,
        "t": t_statistics, "p": p_values, "r_squared": r_squared,
        "df_residual": observations - parameters, "n": observations,
    }


def regress(subset):
    """Straight line through one group, with the correlation that goes with it."""
    fit = stats.linregress(subset[KM].to_numpy(float), subset[KCAT].to_numpy(float))
    return {
        "n": len(subset), "slope": fit.slope, "intercept": fit.intercept,
        "r": fit.rvalue, "r_squared": fit.rvalue ** 2, "p": fit.pvalue,
        "slope_error": fit.stderr,
    }


def plot(table, fits):
    style()
    figure, axes = plt.subplots(figsize=(3.5, 3.0))

    for carries, colour, label in ((False, WITHOUT_COLOUR, f"without {SUBSTITUTION}"),
                                   (True, WITH_COLOUR, f"with {SUBSTITUTION}")):
        subset = table[table["carries_S68P"] == carries]
        axes.errorbar(subset[KM], subset[KCAT],
                      xerr=subset[KM_ERROR], yerr=subset[KCAT_ERROR],
                      fmt="none", ecolor=colour, elinewidth=0.5, alpha=0.5, zorder=1)
        axes.scatter(subset[KM], subset[KCAT], s=16,
                     facecolors="none" if not carries else colour,
                     edgecolors=colour, linewidths=0.9, label=label, zorder=2)

        fit = fits[carries]
        grid = np.linspace(subset[KM].min(), subset[KM].max(), 2)
        axes.plot(grid, fit["intercept"] + fit["slope"] * grid,
                  color=colour, linewidth=1.0, zorder=3)

    handles, labels = axes.get_legend_handles_labels()
    order = [labels.index(f"with {SUBSTITUTION}"), labels.index(f"without {SUBSTITUTION}")]

    axes.set_xlabel("$K_M$ (mM)")
    axes.set_ylabel("$k_{cat}$ (s$^{-1}$)")
    axes.set_xlim(0, np.ceil(table[KM].max()) + 1)
    axes.set_ylim(0, 135)

    axes.legend([handles[i] for i in order], [labels[i] for i in order],
                loc="upper left", fontsize=8, handletextpad=0.4,
                borderpad=0.0, labelspacing=0.3)

    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(THIS_DIR / f"kcat_vs_KM_by_{SUBSTITUTION}.{extension}", dpi=400)
    plt.close(figure)


def write_summary(table, model, fits):
    lines = [
        "kcat against KM, split by whether the variant carries S68P",
        "on S7's set: kinetics measured and a high-confidence fitness score",
        "",
        f"variants: {len(table)}",
        f"  carrying {SUBSTITUTION}:     {int(table['carries_S68P'].sum())}",
        f"  without {SUBSTITUTION}:      {int((~table['carries_S68P']).sum())}",
        "",
        "Straight line through each group",
        f"{'group':16s} {'n':>4} {'slope':>9} {'se':>8} {'r':>7} {'R2':>7} {'p':>10}",
    ]
    for carries, label in ((True, f"with {SUBSTITUTION}"), (False, f"without {SUBSTITUTION}")):
        fit = fits[carries]
        lines.append(f"{label:16s} {fit['n']:>4} {fit['slope']:>9.3f} {fit['slope_error']:>8.3f} "
                     f"{fit['r']:>7.3f} {fit['r_squared']:>7.3f} {fit['p']:>10.2e}")

    lines += [
        "",
        "ANCOVA: kcat ~ KM + S68P + KM x S68P",
        f"{'term':12s} {'estimate':>11} {'se':>10} {'t':>8} {'p':>11}",
    ]
    for name, estimate, error, t, p in zip(model["names"], model["coefficients"],
                                           model["errors"], model["t"], model["p"]):
        lines.append(f"{name:12s} {estimate:>11.4f} {error:>10.4f} {t:>8.3f} {p:>11.3e}")
    lines += [
        "",
        f"model R2 = {model['r_squared']:.3f} on {model['df_residual']} residual degrees of freedom",
        f"interaction p = {model['p'][3]:.3e}",
        "",
        "For the caption",
        f"kcat rises with KM across the variants carrying {SUBSTITUTION} "
        f"(slope {fits[True]['slope']:.1f} s-1 mM-1, r = {fits[True]['r']:.2f}, "
        f"p = {fits[True]['p']:.3g}, n = {fits[True]['n']}) but not across those without it "
        f"(slope {fits[False]['slope']:.1f} s-1 mM-1, r = {fits[False]['r']:.2f}, "
        f"p = {fits[False]['p']:.2g}, n = {fits[False]['n']}); the two slopes differ "
        f"(ANCOVA interaction p = {model['p'][3]:.3g}).",
    ]
    text = "\n".join(lines)
    (THIS_DIR / "ancova_summary.txt").write_text(text + "\n")
    print(text)


def main():
    table = load()
    model = ancova(table)
    fits = {
        True: regress(table[table["carries_S68P"]]),
        False: regress(table[~table["carries_S68P"]]),
        "interaction_p": model["p"][3],
    }
    table.to_csv(THIS_DIR / "kinetics_by_S68P.csv", index=False)
    plot(table, fits)
    write_summary(table, model, fits)


if __name__ == "__main__":
    raise SystemExit(main())
