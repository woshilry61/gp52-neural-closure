# Publication figure package

These figures visualize the completed unified independent-CFD baseline experiment. No model, prediction, inversion, split or metric definition is changed.

All figures are 180 mm wide. Body labels are 8 pt, tick labels 7.5 pt. PDFs contain vector graphics with embedded TrueType fonts; SVG text remains editable. PNG files are exported at 600 dpi. Preview images are rendered from the actual PDFs with Poppler at 160 dpi. No target journal has been specified; adapt width and typeface to its instructions before submission.

The first five figures form the main set. S1-S6 show all three sampling locations for each topology. all-figures.pdf contains the 11 figure pages. captions.txt provides complete English captions separately from the graphics.

Illustrative traces use the first test ID (80), seed 42, and noise realization 1 for inversion. This is an explicit example, not a claim that it is the median case. All 616 scoring samples are drawn. Summary figures use all 32 held-out inlet cases. Case-level bootstrap intervals average noise repetitions and training-seed errors within each case first. They do not measure full experimental, CFD-discretization or new-blood-flow uncertainty.

Figure data are supplied as CSV and NPZ. per-case-errors.csv is an exact copy of the locked evaluation table; plotted-statistics.csv contains the plotted means and intervals. source-data-case80.npz contains all locations and observation repetitions for the example. The time-series CSV contains known-inlet and three-site noisy-inverse traces; the inlet CSV includes one-, two- and three-site reconstruction.

From the repository root, regenerate with:

    .venv/bin/python -m gp52.unified.paper_figures

Requirements are the existing experiment outputs, NumPy, Matplotlib and Poppler (pdftoppm, pdfinfo, pdffonts). The included Python file is the exact plotting source; it runs as the repository module above and uses the frozen experiment files.
