"""Publication figures from frozen predictions; no model fitting or rescoring.

Run: .venv/bin/python -m gp52.unified.paper_figures
The first test ID and the first declared training seed are used mechanically
for illustrative traces. Population statistics use all 32 held-out inlet cases.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import html
import json
import shutil
import subprocess
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

from .protocol import ROOT, RUN, OUT, TEST, plan

DEST = OUT / "paper-figures"
CASE_ID = int(TEST[0])
SEED = 42
NOISE_REP = 1
LAYOUT = 3
NETWORKS = ("single", "three", "seven")
NAMES = {"single": "Single vessel", "three": "Three vessels", "seven": "Seven vessels"}
SPECIES = ("Th", "Fg", "Fn")
WIDTH = 180 / 25.4
COLORS = {"reference": "#222222", "C0": "#D55E00", "new": "#0072B2"}
STYLES = {"reference": "-", "C0": (0, (5, 2, 1, 2)), "new": (0, (4, 2))}
LABELS = {"reference": "CFD-based reference", "C0": "C0", "new": "Neural baseline"}
MARKERS = {"C0": "s", "new": "o"}
LOCATIONS = {
    "single": ("x = 30 mm", "x = 6 mm", "x = 54 mm"),
    "three": ("V1, x = 30 mm", "V2, x = 10 mm", "V3, x = 17 mm"),
    "seven": ("V1, x = 30 mm", "V4, x = 19.5 mm", "V7, x = 47 mm"),
}
BOOTSTRAP_SEED = 52703103
N_BOOTSTRAP = 3000
RECORDS = []
BOUND_CHECKS = []
TIME_SOURCE = []
INLET_SOURCE = []

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.linewidth": .65,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": .6,
    "ytick.major.width": .6,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 1.15,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})


def load_data():
    with np.load(RUN / "report-curves.npz") as archive:
        d = {k: archive[k] if k in ("t", "knots") else archive[k][CASE_ID - 80]
             for k in archive.files}
    for network in NETWORKS:
        with np.load(RUN / network / "observations.npz") as z:
            d[network + "-observations"] = z["observations"][CASE_ID]
    d["observation_times"] = np.array(plan()["observation_times"])
    np.savez_compressed(DEST / "source-data-case80.npz", **d)
    return d


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def statistics():
    raw = list(csv.DictReader((OUT / "per-case-errors.csv").open()))
    groups = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r["family"] not in ("C0", "new"):
            continue
        setting = r["setting"]
        if setting == "inverse" and int(r["rep"]) == 0:
            continue
        for metric in SPECIES if setting == "forward" else ("inlet",) + SPECIES:
            key = (r["network"], setting, int(r["sites"]), metric, r["family"])
            groups[key][int(r["case"])].append(float(r[metric]))
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(TEST), size=(N_BOOTSTRAP, len(TEST)))
    rows = []
    stats = {}
    for key, by_case in sorted(groups.items()):
        assert sorted(by_case) == list(TEST)
        values = 100 * np.array([np.mean(by_case[int(case)]) for case in TEST])
        samples = values[bootstrap].mean(axis=1)
        low, high = np.quantile(samples, [.025, .975])
        mean = values.mean()
        stats[key] = (mean, low, high)
        rows.append(dict(network=key[0], setting=key[1], sites=key[2],
                         metric=key[3], method=key[4], n_cases=len(TEST),
                         mean_percent=mean, ci_low_percent=low,
                         ci_high_percent=high))
    # Check plotted means against the locked report, without rescoring a model.
    summary = list(csv.DictReader((OUT / "summary.csv").open()))
    for r in summary:
        if r["family"] not in ("C0", "new") or r["noise"] == "clean":
            continue
        for metric in SPECIES if r["setting"] == "forward" else ("inlet",) + SPECIES:
            key = (r["network"], r["setting"], int(r["sites"]), metric, r["family"])
            assert abs(stats[key][0] - 100 * float(r[metric])) < 1e-10
    write_csv(DEST / "plotted-statistics.csv", rows)
    shutil.copy2(OUT / "per-case-errors.csv", DEST / "per-case-errors.csv")
    shutil.copy2(OUT / "paired.csv", DEST / "paired-differences.csv")
    return stats


def axis_style(ax, letter, time_axis=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", pad=2.5)
    ax.text(0, 1.035, "(" + letter + ")", transform=ax.transAxes,
            fontweight="bold", fontsize=8.5, va="bottom", ha="left")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    if time_axis:
        ax.set_xlim(0, 1.23)
        ax.set_xticks([0, .4, .8, 1.2])
    ax.set_axisbelow(True)


def legend(fig, observed=False, inlet=False):
    handles = [Line2D([], [], color=COLORS[m], linestyle=STYLES[m], lw=1.25,
                      label=("True inlet" if inlet else LABELS[m])
                      if m == "reference" else LABELS[m])
               for m in ("reference", "C0", "new")]
    if observed:
        handles.append(Line2D([], [], ls="", marker="o", markersize=3,
                              markerfacecolor="white", markeredgecolor="#666666",
                              markeredgewidth=.65, label="Th/Fg observations"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1),
               ncol=len(handles), handlelength=3, columnspacing=1.5)


def series(d, network, mode, species, site):
    data = {"reference": d[network + "-truth"][species, site]}
    for method, key in (("C0", "c0"), ("new", "nn")):
        arr = d[network + "-" + key + "-" + mode]
        data[method] = arr[species, site] if mode == "forward" else \
            arr[NOISE_REP, LAYOUT - 1, species, site]
    return data


def save_figure(fig, stem, title, caption, bundle):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                outside.append(item.get_text())
    assert not outside, (stem, outside)
    BOUND_CHECKS.append(dict(figure=stem, text_within_page=True))
    metadata = {"Title": title, "Creator": "GP52", "CreationDate": None, "ModDate": None}
    fig.savefig(DEST / (stem + ".pdf"), metadata=metadata)
    fig.savefig(DEST / (stem + ".svg"),
                metadata={"Title": title, "Creator": "GP52", "Date": None})
    svg = DEST / (stem + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(DEST / (stem + ".png"), dpi=600,
                metadata={"Software": "GP52 figure pipeline", "Description": caption})
    bundle.savefig(fig)
    subprocess.run(["pdftoppm", "-r", "160", "-png", "-singlefile",
                    str(DEST / (stem + ".pdf")), str(DEST / "preview" / stem)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    RECORDS.append(dict(stem=stem, title=title, caption=caption,
                        width_mm=180, height_mm=fig.get_figheight() * 25.4))
    plt.close(fig)
    print("SAVED", stem, flush=True)


def time_panels(d, mode, bundle, network=None):
    """Three species by three networks, or three predeclared sites."""
    fig, axs = plt.subplots(3, 3, figsize=(WIDTH, 5.65), sharex=True, sharey="row")
    fig.subplots_adjust(left=.10, right=.985, bottom=.105, top=.90,
                        wspace=.18, hspace=.26)
    panels = [(n, 0) for n in NETWORKS] if network is None else [(network, p) for p in range(3)]
    for s, species in enumerate(SPECIES):
        all_values = []
        for col, (name, site) in enumerate(panels):
            ax = axs[s, col]
            axis_style(ax, chr(97 + 3*s + col))
            for method, y in series(d, name, mode, s, site).items():
                ax.plot(d["t"], y, color=COLORS[method], linestyle=STYLES[method],
                        lw=1.2 if method == "reference" else 1.1,
                        zorder=2 if method == "C0" else 3)
                all_values.extend(y)
            if mode == "inverse" and s < 2:
                obs = d[name + "-observations"][NOISE_REP, s, site]
                ax.scatter(d["observation_times"], obs, s=8, facecolors="white",
                           edgecolors="#666666", linewidths=.55, zorder=4)
                all_values.extend(obs)
            if s == 0:
                # One header prevents long location names from touching panel letters.
                ax.texts[-1].remove()
                heading = NAMES[name] if network is None else LOCATIONS[name][site]
                ax.set_title("(" + chr(97 + col) + ")  " + heading, loc="left", pad=7)
            if col == 0:
                ax.set_ylabel(species + " (normalized)")
            if s == 2:
                ax.set_xlabel("Time (s)")
        lower = min(0., min(all_values))
        if lower > -1e-8:
            lower = 0.
        upper = max(all_values)
        pad = .06 * max(upper - lower, .05)
        axs[s, 0].set_ylim(lower - pad if lower < 0 else 0, upper + pad)
    legend(fig, observed=mode == "inverse")
    footer = "Known inlet" if mode == "forward" else "3 sites; noise realization 1"
    fig.text(.985, .015, f"Test case {CASE_ID}; neural seed {SEED}; {footer}",
             ha="right", fontsize=7, color="#444444")
    if network is None:
        stem = "fig01_forward_time" if mode == "forward" else "fig03_inverse_time"
        title = "Known-inlet concentration trajectories" if mode == "forward" else \
                "Concentration trajectories after sparse inverse reconstruction"
        locations = "The columns show single-, three- and seven-vessel networks at x = 30 mm in the inlet vessel."
    else:
        number = 1 + 2 * NETWORKS.index(network) + (mode == "inverse")
        stem = f"figS{number}_{network}_{mode}"
        title = f"{NAMES[network]}: " + ("known-inlet trajectories" if mode == "forward" else "inverse reconstruction")
        locations = "The columns show the three predeclared sampling sites: " + "; ".join(LOCATIONS[network]) + ". Vessel labels are one-based in model order."
    caption = (locations + " Rows show cross-section-averaged Th, Fg and Fn in normalized model units. "
               "Black solid curves are the independently generated CFD-based biochemical reference; "
               "orange dash-dot curves are fixed C0; blue dashed curves are the neural baseline. "
               f"Illustrative test case {CASE_ID} is the first held-out ID, selected without performance ranking. "
               f"Neural trajectories use seed {SEED}, not a seed-averaged prediction. "
               "Every stored 2 ms scoring sample over 0-1.23 s is plotted without smoothing. "
               "Axes share the same concentration range within each species row. ")
    if mode == "inverse":
        caption += ("Both methods infer seven inlet amplitudes from the same three-site Th/Fg observations "
                    "with Gaussian noise of absolute standard deviation 0.02 (realization 1). "
                    "Hollow circles are those observations; Fn is unobserved. Negative noisy Th observations, "
                    "where present, are retained. ")
    else:
        caption += "Both methods receive the prescribed inlet and the same independent mean blood fields. "
    caption += "Each topology has its own fixed blood condition and separately trained neural weights."
    save_figure(fig, stem, title, caption, bundle)


def inlet_panels(d, bundle):
    fig, axs = plt.subplots(3, 3, figsize=(WIDTH, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.105, top=.90,
                        wspace=.18, hspace=.28)
    for row, layout in enumerate((1, 2, 3)):
        for col, network in enumerate(NETWORKS):
            ax = axs[row, col]
            axis_style(ax, chr(97 + 3*row + col), time_axis=False)
            ax.set_xlim(0, .82)
            ax.set_ylim(0, 1.2)
            ax.set_xticks([0, .2, .4, .6, .8])
            ax.set_yticks([0, .4, .8, 1.2])
            curves = {"reference": d["inlet-truth"],
                      "C0": d[network + "-c0-inlet"][NOISE_REP, layout-1],
                      "new": d[network + "-nn-inlet"][NOISE_REP, layout-1]}
            for method, a in curves.items():
                ax.plot(d["knots"], a, color=COLORS[method], linestyle=STYLES[method],
                        lw=1.2, marker=".", markersize=3)
                for k, (t, v) in enumerate(zip(d["knots"], a)):
                    INLET_SOURCE.append(dict(network=network,case=CASE_ID,seed=SEED,
                        noise_rep=NOISE_REP,sites=layout,method=method,knot=k,time_s=t,amplitude=v))
            if row == 0:
                ax.set_title(NAMES[network], pad=7)
            if col == 0:
                count = "1 site" if layout == 1 else f"{layout} sites"
                ax.set_ylabel(count + "\nInlet Th (normalized)")
            if row == 2:
                ax.set_xlabel("Time (s)")
    legend(fig, inlet=True)
    fig.text(.985, .015, f"Test case {CASE_ID}; neural seed {SEED}; noise realization 1",
             ha="right", fontsize=7, color="#444444")
    caption = (
        "Columns show the single-, three- and seven-vessel networks; rows use one, two and three "
        "nested measurement sites. The black curve is the prescribed seven-knot Th inlet. "
        "C0 (orange dash-dot) and the neural baseline (blue dashed) infer the same seven amplitudes "
        "from paired noisy Th/Fg observations, with noise realization 1 and standard deviation 0.02. "
        "Dots indicate parameter knots, not additional observations. All axes use the same limits. "
        f"Test case {CASE_ID} is the first held-out ID; the neural curves use seed {SEED}. "
        "Only the scored inlet interval 0-0.82 s is shown; the final amplitude is held constant "
        "during the remaining concentration simulation to 1.23 s. The temporal inlet is prescribed, "
        "not measured from CFD. The additional sites lie at different downstream distances in "
        "different topologies, so this is not a controlled vessel-count-only comparison.")
    save_figure(fig, "fig02_inlet_recovery", "Sparse reconstruction of the inlet waveform", caption, bundle)


def error_legend(fig):
    handles = [Line2D([], [], color=COLORS[m], linestyle=STYLES[m],
                      marker=MARKERS[m], markersize=4, label=LABELS[m], lw=1.1)
               for m in ("C0", "new")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1),
               ncol=2, handlelength=3, columnspacing=2)


def errorbar(ax, x, values, method, connected):
    a = np.asarray(values)
    ax.errorbar(x, a[:, 0], yerr=np.vstack((a[:, 0]-a[:, 1], a[:, 2]-a[:, 0])),
                color=COLORS[method], marker=MARKERS[method], markersize=3.6,
                linestyle=STYLES[method] if connected else "none",
                lw=1.05, elinewidth=.8, capsize=2, capthick=.75, zorder=3)


def summary_figures(stats, bundle):
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.65))
    fig.subplots_adjust(left=.09, right=.985, bottom=.23, top=.78, wspace=.32)
    for s, metric in enumerate(SPECIES):
        ax = axs[s]
        axis_style(ax, chr(97+s), time_axis=False)
        upper = 0
        for method, offset in (("C0", -.10), ("new", .10)):
            values = [stats[(n, "forward", 0, metric, method)] for n in NETWORKS]
            upper = max(upper, max(v[2] for v in values))
            errorbar(ax, np.arange(3)+offset, values, method, False)
        ax.set_title(metric, pad=7)
        ax.set_xticks(np.arange(3), ["Single", "Three", "Seven"])
        ax.set_xlim(-.4, 2.4)
        ax.set_ylim(0, upper*1.12)
        ax.set_xlabel("Vessel network")
        if s == 0:
            ax.set_ylabel("Relative " + r"$L^2$" + " error (%)")
    error_legend(fig)
    fig.text(.985, .015, "32 test inlet cases; mean and 95% case-bootstrap CI",
             ha="right", fontsize=7, color="#444444")
    caption = (
        "Known-inlet whole-network errors for Th, Fg and Fn. Symbols are mean relative L2 errors "
        "over 32 held-out inlet cases, weighted within each case by reference volume and trapezoidal "
        "time weights over 0-1.23 s. Neural errors are averaged across seeds 42, 43 and 44 within "
        "each case before averaging cases; this is not the error of an ensemble prediction. "
        "Error bars are 95% percentile bootstrap confidence intervals for the case mean "
        "(3,000 resamples of the 32 complete inlet cases; RNG seed 52703103). "
        "Seeds and time samples are not treated as additional independent cases. "
        "The three topologies have different fixed geometries and blood conditions.")
    save_figure(fig, "fig04_forward_errors", "Known-inlet test errors with uncertainty", caption, bundle)

    metrics = ("inlet",) + SPECIES
    fig, axs = plt.subplots(3, 4, figsize=(WIDTH, 5.95), sharex=True, sharey="col")
    fig.subplots_adjust(left=.10, right=.985, bottom=.105, top=.90,
                        wspace=.20, hspace=.30)
    for col, metric in enumerate(metrics):
        upper = max(stats[(n, "inverse", k, metric, m)][2]
                    for n in NETWORKS for k in (1,2,3) for m in ("C0","new"))
        for row, network in enumerate(NETWORKS):
            ax = axs[row, col]
            axis_style(ax, chr(97 + row*4 + col), time_axis=False)
            for method, offset in (("C0", -.035), ("new", .035)):
                values = [stats[(network, "inverse", k, metric, method)] for k in (1,2,3)]
                errorbar(ax, np.arange(1,4)+offset, values, method, True)
            ax.set_xlim(.7, 3.3)
            ax.set_xticks([1,2,3])
            ax.set_ylim(0, upper*1.10)
            if row == 0:
                ax.set_title("Inlet" if metric == "inlet" else metric, pad=7)
            if col == 0:
                ax.set_ylabel(NAMES[network] + "\nRelative " + r"$L^2$" + " error (%)")
            if row == 2:
                ax.set_xlabel("Measurement sites")
    error_legend(fig)
    fig.text(.985, .015, "32 test inlet cases; mean and 95% case-bootstrap CI",
             ha="right", fontsize=7, color="#444444")
    caption = (
        "Noisy sparse-inverse errors. Rows show single-, three- and seven-vessel networks. "
        "Columns show inlet recovery and reconstructed Th, Fg and unobserved Fn. "
        "Th/Fg are observed at 24 times using one, two or three nested sites. "
        "Noise has absolute standard deviation 0.02. For each inlet case, errors are averaged "
        "over the three paired noise realizations and, for the neural baseline, three training "
        "seeds; the 32 case means then receive equal weight. Error bars are 95% percentile "
        "bootstrap confidence intervals using 3,000 resamples of complete inlet cases. "
        "Inlet errors use 0-0.82 s and concentration errors use 0-1.23 s. "
        "Each column shares the same vertical limits across the three topologies, with zero "
        "included. Small horizontal offsets separate overlapping intervals. Connections between "
        "the three discrete site counts only guide the eye. Marginal intervals are descriptive; "
        "paired method-difference intervals are supplied in paired-differences.csv.")
    save_figure(fig, "fig05_inverse_errors", "Sparse-inverse test errors with uncertainty", caption, bundle)


def source_tables(d):
    for network in NETWORKS:
        for mode in ("forward", "inverse"):
            for species, name in enumerate(SPECIES):
                for site in range(3):
                    for method, values in series(d, network, mode, species, site).items():
                        for t, value in zip(d["t"], values):
                            TIME_SOURCE.append(dict(network=network,case=CASE_ID,seed=SEED,
                                mode=mode,sites=0 if mode=="forward" else LAYOUT,
                                noise_rep=0 if mode=="forward" else NOISE_REP,
                                species=name,site_index=site+1,method=method,
                                time_s=t,concentration=value))
    write_csv(DEST / "time-series-case80.csv", TIME_SOURCE)
    write_csv(DEST / "inlet-reconstruction-case80.csv", INLET_SOURCE)


def documents_and_checks():
    caption_text = "\n\n".join(r["stem"] + "\n" + r["caption"] for r in RECORDS) + "\n"
    (DEST / "captions.txt").write_text(caption_text)
    shutil.copy2(Path(__file__), DEST / "paper_figures.py")
    readme = (
        "# Publication figure package\n\n"
        "These figures visualize the completed unified independent-CFD baseline experiment. "
        "No model, prediction, inversion, split or metric definition is changed.\n\n"
        "All figures are 180 mm wide. Body labels are 8 pt, tick labels 7.5 pt. "
        "PDFs contain vector graphics with embedded TrueType fonts; SVG text remains editable. "
        "PNG files are exported at 600 dpi. Preview images are rendered from the actual PDFs "
        "with Poppler at 160 dpi. No target journal has been specified; adapt width and typeface "
        "to its instructions before submission.\n\n"
        "The first five figures form the main set. S1-S6 show all three sampling locations "
        "for each topology. all-figures.pdf contains the 11 figure pages. captions.txt provides "
        "complete English captions separately from the graphics.\n\n"
        "Illustrative traces use the first test ID (80), seed 42, and noise realization 1 "
        "for inversion. This is an explicit example, not a claim that it is the median case. "
        "All 616 scoring samples are drawn. Summary figures use all 32 held-out inlet cases. "
        "Case-level bootstrap intervals average noise repetitions and training-seed errors "
        "within each case first. They do not measure full experimental, CFD-discretization "
        "or new-blood-flow uncertainty.\n\n"
        "Figure data are supplied as CSV and NPZ. per-case-errors.csv is an exact copy of the "
        "locked evaluation table; plotted-statistics.csv contains the plotted means and intervals. "
        "source-data-case80.npz contains all locations and observation repetitions for the example. "
        "The time-series CSV contains known-inlet and three-site noisy-inverse traces; "
        "the inlet CSV includes one-, two- and three-site reconstruction.\n\n"
        "From the repository root, regenerate with:\n\n"
        "    .venv/bin/python -m gp52.unified.paper_figures\n\n"
        "Requirements are the existing experiment outputs, NumPy, Matplotlib and Poppler "
        "(pdftoppm, pdfinfo, pdffonts). The included Python file is the exact plotting source; "
        "it runs as the repository module above and uses the frozen experiment files.\n")
    (DEST / "README.md").write_text(readme)
    for record in RECORDS:
        pdf = DEST / (record["stem"] + ".pdf")
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        assert "Pages:           1" in info
        assert "Type 3" not in fonts
        record["pdfinfo"] = info
        record["pdffonts"] = fonts
    (DEST / "render-checks.json").write_text(json.dumps(
        dict(text_bounds=BOUND_CHECKS, figures=RECORDS), indent=2))
    items = []
    for r in RECORDS:
        stem = r["stem"]
        links = " · ".join(f'<a href="{stem}.{ext}">{label}</a>'
                          for ext,label in (("pdf","Vector PDF"),("svg","Editable SVG"),("png","600 dpi PNG")))
        items.append(f'<section><h2>{html.escape(r["title"])}</h2>'
                     f'<p class="downloads">{links}</p>'
                     f'<img src="preview/{stem}.png" alt="{html.escape(r["title"])}" loading="lazy">'
                     f'<p class="caption"><b>{stem}.</b> {html.escape(r["caption"])}</p></section>')
    page = """<!doctype html><meta charset="utf-8"><title>GP52 publication figures</title>
<style>body{font:16px system-ui;line-height:1.6;max-width:1060px;margin:35px auto;padding:0 24px;color:#182b37;background:#f7f8fa}h1{font-size:28px;line-height:1.25}h2{font-size:21px}a{color:#005f91}section{background:white;border:1px solid #dbe2e8;padding:22px;margin:28px 0}img{width:100%;height:auto;display:block}.caption{font-size:14px;color:#3c4c57}.downloads{font-size:15px}</style>
<h1>GP52: publication figures</h1>
<p>论文排版用图。数据与模型保持不变；时间曲线采用明确标注的测试案例，统计图使用全部 32 个测试案例。</p>
<p><a href="all-figures.pdf">全部 11 页 PDF</a> · <a href="publication-figures.zip">下载完整图包</a> · <a href="captions.txt">English figure captions</a> · <a href="../summary-zh.html">完整结果报告</a></p>
<p>PDF/SVG 为矢量图，PNG 为 600 dpi。预览从 PDF 渲染。下列示例曲线采用测试 ID 80 和种子 42；统计图的误差条为按入口案例重采样得到的 95% 置信区间。</p>
""" + "\n".join(items)
    (DEST / "index.html").write_text(page)
    inputs = [RUN / "report-curves.npz", OUT / "per-case-errors.csv", OUT / "summary.csv",
              OUT / "paired.csv", OUT / "manifest.json", Path(__file__)]
    inputs += [RUN / n / "observations.npz" for n in NETWORKS]
    manifest = dict(case_id=CASE_ID, case_selection="First held-out ID; no performance ranking",
                    neural_trace_seed=SEED, inverse_trace_noise_rep=NOISE_REP,
                    bootstrap_resamples=N_BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED,
                    independent_test_cases=len(TEST), png_dpi=600, figure_width_mm=180,
                    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(DEST / "publication-figures.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(DEST.rglob("*")):
            if p.is_file() and p.name != "publication-figures.zip":
                archive.write(p, p.relative_to(DEST))
    print("COMPLETE", len(RECORDS), "figures", flush=True)


def main():
    DEST.mkdir(exist_ok=True)
    (DEST / "preview").mkdir(exist_ok=True)
    d = load_data()
    stats = statistics()
    with PdfPages(DEST / "all-figures.pdf",
                  metadata={"Title": "GP52 unified baseline figures", "Creator": "GP52",
                            "CreationDate": None, "ModDate": None}) as bundle:
        time_panels(d, "forward", bundle)
        inlet_panels(d, bundle)
        time_panels(d, "inverse", bundle)
        summary_figures(stats, bundle)
        for network in NETWORKS:
            time_panels(d, "forward", bundle, network)
            time_panels(d, "inverse", bundle, network)
    source_tables(d)
    documents_and_checks()


if __name__ == "__main__":
    main()
