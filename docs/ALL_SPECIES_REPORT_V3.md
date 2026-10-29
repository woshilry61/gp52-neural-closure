# Complete species reporting, v3

Use reports/all-species-v3/report.html for the readable report and
reports/all-species-v3/curves.html for the standalone, offline curve viewer.
Both HTML pages are generated from the frozen evaluation records; the viewer
embeds all cases and all noise realizations without external libraries.

The primary inverse concentration errors now include Th, Fg and Fn, all over
0–1.23 s and every cell. Inlet error remains over 0–0.82 s. Observation windows,
noise arrays, sensor layouts, inverse fits, validation regularization choices,
C0 definitions and neural weights are unchanged. Known-inlet forward replay is
performed separately using the frozen GPU implementations.

The viewer displays all three species at all three fixed spatial sites. In
single-vessel runs these are 6, 30 and 54 mm. In three-vessel runs they are the
parent midpoint and the two daughter midpoints. Observation markers appear
only on Th/Fg at the sites actually included in a selected inverse layout.
All field-error tables use the full spatial domain, not only the plotted points.

Run the appropriate command from this checkout and its existing environment:

    .venv/bin/python scripts/report_all_species.py --network three

The network flag matches this checkout. Cached known-inlet replays are reused. No neural training,
inverse optimization or regularization selection is invoked.

The complete float64 time curves and identity metadata are saved in
runs/all-species-v3/all-concentration-curves.npz. The HTML embeds float32 display
copies only; errors are calculated from the original float64 full fields.
All figures, clean/noisy aggregate tables, per-case/per-vessel errors, paired
case bootstrap intervals and preservation checks are saved beside the report.

The legacy v2 reports and model locks remain intact. They retain the same inlet
and Fn point estimates; this version adds the missing species and fuller display.
The scripts are identical in both checkouts, with an explicit geometry selector.
