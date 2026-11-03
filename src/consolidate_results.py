"""Create reporting artifacts from saved JSON; never trains models or alters raw outputs."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LANGUAGES = ("english", "hindi", "nepali")
REPLICATES = ("replicate_1", "replicate_2", "replicate_3")
COLORS = ("#2563eb", "#d97706", "#059669")
REGIME_COLORS = ("#2563eb", "#c2410c")


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def penalty(dedicated, shared):
    return (shared - dedicated) / dedicated * 100


def metrics(models, language):
    return (models[f"{language}_separate"]["test"][language]["metrics"],
            models["shared_multilingual"]["test"][language]["metrics"])


def svg(title, subtitle, body, height=680):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img"><title>{title}</title><style>text{{font-family:Arial,sans-serif;fill:#172033}}.t{{font-size:26px;font-weight:700}}.s{{font-size:15px;fill:#526070}}.a{{font-size:13px;fill:#526070}}.l{{font-size:14px}}.g{{stroke:#d9e0ea}}.x{{stroke:#64748b;stroke-width:1.5}}</style><rect width="100%" height="100%" fill="white"/><text x="110" y="48" class="t">{title}</text><text x="110" y="76" class="s">{subtitle}</text>{body}</svg>'''


def chart_axes(y_min, y_max, step=2):
    y = lambda value: 145 + (y_max - value) * 415 / (y_max - y_min)
    body = '<text x="110" y="102" class="s">Penalty = (shared tokens − dedicated tokens) / dedicated tokens × 100; positive means more tokens.</text>'
    for tick in range(y_min, y_max + 1, step):
        body += f'<line x1="110" y1="{y(tick):.1f}" x2="1030" y2="{y(tick):.1f}" class="g"/><text x="94" y="{y(tick)+5:.1f}" class="a" text-anchor="end">{tick}%</text>'
    return body + f'<line x1="110" y1="145" x2="110" y2="560" class="x"/><line x1="110" y1="560" x2="1030" y2="560" class="x"/>', y


def write_figures(sweep, replicate_values, summary):
    FIGURES.mkdir(exist_ok=True)
    body, y = chart_axes(12, 21)
    xs = (110, 570, 1030)
    label_offsets = {
        "english": (12, 20, "end"),
        "hindi": (-12, -12, "end"),
        "nepali": (-12, -12, "end"),
    }
    for language, color in zip(LANGUAGES, COLORS):
        vals = []
        for size in (512, 1024, 2048):
            dedicated, shared = metrics(sweep["models"][str(size)], language)
            vals.append(penalty(dedicated["tokens"], shared["tokens"]))
        body += f'<polyline points="{" ".join(f"{x},{y(v):.1f}" for x, v in zip(xs, vals))}" fill="none" stroke="{color}" stroke-width="3"/>'
        for index, (x, value) in enumerate(zip(xs, vals)):
            x_offset, y_offset, anchor = label_offsets[language] if index == 2 else (0, -12, "middle")
            body += f'<circle cx="{x}" cy="{y(value):.1f}" r="5" fill="{color}"/><text x="{x+x_offset}" y="{y(value)+y_offset:.1f}" class="l" text-anchor="{anchor}">{value:.2f}%</text>'
    for x, size in zip(xs, (512, 1024, 2048)):
        body += f'<text x="{x}" y="590" class="a" text-anchor="middle">{size}</text>'
    for index, (language, color) in enumerate(zip(LANGUAGES, COLORS)):
        legend_y = 122 + index * 22
        body += f'<line x1="760" y1="{legend_y}" x2="784" y2="{legend_y}" stroke="{color}" stroke-width="3"/><circle cx="772" cy="{legend_y}" r="4" fill="{color}"/><text x="794" y="{legend_y+5}" class="l">{language.title()}</text>'
    body += '<text x="570" y="625" class="a" text-anchor="middle">Requested vocabulary size</text><text x="110" y="655" class="a">Single deterministic balanced run per size; no resampling uncertainty is represented.</text>'
    (FIGURES / "shared_vocabulary_penalty_sweep.svg").write_text(svg("Shared-vocabulary token-count penalty across vocabulary sizes", "Single-run balanced vocabulary sweep; fixed ~300 KB budget per language.", body), encoding="utf-8")
    body, y = chart_axes(12, 21)
    centers = (265, 570, 875)
    shapes = ("circle", "square", "triangle")
    for center, language, color in zip(centers, LANGUAGES, COLORS):
        mean, sd = summary[language]
        body += f'<line x1="{center}" y1="{y(mean-sd):.1f}" x2="{center}" y2="{y(mean+sd):.1f}" stroke="#111827" stroke-width="3"/><rect x="{center-6}" y="{y(mean)-6:.1f}" width="12" height="12" fill="#111827"/>'
        for offset, value, shape in zip((-28, 0, 28), replicate_values[language], shapes):
            x_position, y_position = center + offset, y(value)
            if shape == "circle":
                body += f'<circle cx="{x_position}" cy="{y_position:.1f}" r="7" fill="{color}" stroke="white" stroke-width="1.5"/>'
            elif shape == "square":
                body += f'<rect x="{x_position-6}" y="{y_position-6:.1f}" width="12" height="12" fill="{color}" stroke="white" stroke-width="1.5"/>'
            else:
                body += f'<polygon points="{x_position},{y_position-8:.1f} {x_position-7},{y_position+6:.1f} {x_position+7},{y_position+6:.1f}" fill="{color}" stroke="white" stroke-width="1.5"/>'
        label_y = y(mean + sd) - 20
        body += f'<rect x="{center-78}" y="{label_y-14:.1f}" width="156" height="20" fill="white"/><text x="{center}" y="{label_y:.1f}" class="l" text-anchor="middle">{mean:.2f}% ± {sd:.2f} pp</text><text x="{center}" y="590" class="l" text-anchor="middle">{language.title()}</text>'
    body += '<circle cx="720" cy="620" r="5" fill="#526070"/><text x="732" y="625" class="a">resample 1</text><rect x="825" y="615" width="10" height="10" fill="#526070"/><text x="842" y="625" class="a">resample 2</text><polygon points="940,614 934,626 946,626" fill="#526070"/><text x="952" y="625" class="a">resample 3</text><text x="110" y="655" class="a">Black square and bar: mean ± sample SD. Overlapping resamples make this descriptive, not inferential.</text>'
    (FIGURES / "shared_vocabulary_penalty_resamples.svg").write_text(svg("Shared-vocabulary penalty across three saved resamples", "Vocabulary 1024; fixed held-out tests; approximately 300 KB training budget per language.", body), encoding="utf-8")


def marker(shape, x_position, y_position, color):
    if shape == "circle":
        return f'<circle cx="{x_position}" cy="{y_position:.1f}" r="6" fill="{color}" stroke="white" stroke-width="1.2"/>'
    if shape == "square":
        return f'<rect x="{x_position-5.5}" y="{y_position-5.5:.1f}" width="11" height="11" fill="{color}" stroke="white" stroke-width="1.2"/>'
    return f'<polygon points="{x_position},{y_position-7:.1f} {x_position-6.5},{y_position+5.5:.1f} {x_position+6.5},{y_position+5.5:.1f}" fill="{color}" stroke="white" stroke-width="1.2"/>'


def write_budget_comparison(fixed_values, fixed_summary, matched_values, matched_summary):
    body, y = chart_axes(0, 24, 4)
    body += '<text transform="rotate(-90 32 352)" x="32" y="352" class="a" text-anchor="middle">Token-count penalty (%)</text>'
    centers = (300, 570, 840)
    regimes = (
        (-64, REGIME_COLORS[0], fixed_values, fixed_summary),
        (64, REGIME_COLORS[1], matched_values, matched_summary),
    )
    shapes = ("circle", "square", "triangle")
    for center, language in zip(centers, LANGUAGES):
        for offset, color, series, summary in regimes:
            mean, sd = summary[language]
            x_position = center + offset
            observed = series[language]
            body += f'<line x1="{x_position}" y1="{y(mean-sd):.1f}" x2="{x_position}" y2="{y(mean+sd):.1f}" stroke="#111827" stroke-width="2.5"/>'
            body += f'<line x1="{x_position-11}" y1="{y(mean):.1f}" x2="{x_position+11}" y2="{y(mean):.1f}" stroke="#111827" stroke-width="2"/>'
            for jitter, value, shape in zip((-18, 0, 18), observed, shapes):
                body += marker(shape, x_position + jitter, y(value), color)
            label = f"{mean:.2f}%"
            label_y = y(max(mean + sd, max(observed))) - 36
            body += f'<rect x="{x_position-32}" y="{label_y-13:.1f}" width="64" height="18" fill="white"/><text x="{x_position}" y="{label_y:.1f}" class="a" text-anchor="middle">{label}</text>'
        body += f'<text x="{center}" y="592" class="l" text-anchor="middle">{language.title()}</text>'
    body += '<rect x="120" y="612" width="22" height="12" fill="#2563eb"/><text x="148" y="623" class="a">~900 KB joint total</text>'
    body += '<rect x="320" y="612" width="22" height="12" fill="#c2410c"/><text x="348" y="623" class="a">~300 KB joint total</text>'
    body += '<line x1="536" y1="610" x2="536" y2="626" stroke="#111827" stroke-width="2.5"/><line x1="528" y1="618" x2="544" y2="618" stroke="#111827" stroke-width="2"/><text x="552" y="623" class="a">mean tick and sample SD</text>'
    body += '<circle cx="128" cy="648" r="5" fill="#526070"/><text x="142" y="653" class="a">resample 1</text><rect x="250" y="643" width="10" height="10" fill="#526070"/><text x="268" y="653" class="a">resample 2</text><polygon points="390,642 384,654 396,654" fill="#526070"/><text x="404" y="653" class="a">resample 3</text><text x="510" y="653" class="a">Printed percentage = three-resample mean</text>'
    body += '<text x="110" y="682" class="a">Replicated summary only. The single-run vocabulary sweep is a separate figure and is not plotted here.</text><text x="110" y="704" class="a">Resamples overlap, so sample SDs describe resample spread only. They are not confidence intervals or significance tests.</text>'
    (FIGURES / "shared_vocabulary_penalty_budget_comparison.svg").write_text(svg("Shared-vocabulary penalty under two total training budgets", "Vocabulary 1024; same fixed held-out tests. Color is the joint-training budget; shape is the resample.", body, 724), encoding="utf-8")


def main():
    sweep, repeated, matched, baseline, saved, saved_matched = (load("balanced_vocabulary_sweep_v1.json"), load("repeated_vocabulary_sharing_v1.json"), load("total_budget_matched_v1.json"), load("multilingual_tokenizer_baseline_v1.json"), load("repeated_vocabulary_sharing_v1_summary.json"), load("total_budget_matched_v1_summary.json"))
    rows, values, replicate_details = [], {language: [] for language in LANGUAGES}, {language: [] for language in LANGUAGES}
    for size, models in sweep["models"].items():
        for language in LANGUAGES:
            dedicated, shared = metrics(models, language)
            for condition, data in (("dedicated", dedicated), ("shared", shared)):
                rows.append(["single_run_sweep", sweep["experiment"], size, "", language, condition, data["tokens"], f'{data["bytes_per_token"]:.6f}', data["roundtrip_exact"], "single deterministic balanced run"])
    for replicate in REPLICATES:
        data = repeated["replicates"][replicate]
        for language in LANGUAGES:
            dedicated, shared = metrics(data["models"], language)
            current = penalty(dedicated["tokens"], shared["tokens"])
            values[language].append(current); replicate_details[language].append((dedicated["tokens"], shared["tokens"]))
            for condition, metric in (("dedicated", dedicated), ("shared", shared)):
                rows.append(["three_resample_summary", repeated["experiment"], 1024, replicate, language, condition, metric["tokens"], f'{metric["bytes_per_token"]:.6f}', metric["roundtrip_exact"], f"shared penalty {current:.2f}%"])
    matched_values, matched_details = {language: [] for language in LANGUAGES}, {language: [] for language in LANGUAGES}
    for replicate in REPLICATES:
        data = matched["replicates"][replicate]
        for condition, model in data["models"].items():
            assert model["actual_vocab_size"] == matched["requested_vocab_size"] == 1024
            for language in LANGUAGES:
                assert model["test"][language]["metrics"]["roundtrip_exact"] is True
        for language in LANGUAGES:
            dedicated, shared = metrics(data["models"], language)
            assert dedicated["roundtrip_exact"] is True and shared["roundtrip_exact"] is True
            current = penalty(dedicated["tokens"], shared["tokens"])
            matched_values[language].append(current); matched_details[language].append((dedicated["tokens"], shared["tokens"]))
            for condition, metric in (("dedicated", dedicated), ("shared", shared)):
                rows.append(["total_budget_matched", matched["experiment"], matched["requested_vocab_size"], replicate, language, condition, metric["tokens"], f'{metric["bytes_per_token"]:.6f}', metric["roundtrip_exact"], f"shared penalty {current:.2f}%"])
    for run in baseline["runs"]:
        metric = run["test"]
        rows.append(["contextual_baseline", baseline["experiment"], run["requested_vocab_size"], "", run["language"], run["tokenizer"], metric["tokens"], f'{metric["bytes_per_token"]:.6f}', metric["roundtrip_exact"], "full-corpus dedicated training"])
    summary = {language: (statistics.mean(items), statistics.stdev(items)) for language, items in values.items()}
    for language, (mean, sd) in summary.items():
        assert abs(mean-saved["per_language"][language]["mean_penalty_percent"]) < 1e-12 and abs(sd-saved["per_language"][language]["sample_stddev_percentage_points"]) < 1e-12
    matched_summary = {language: (statistics.mean(items), statistics.stdev(items)) for language, items in matched_values.items()}
    for language, (mean, sd) in matched_summary.items():
        assert abs(mean-saved_matched["per_language"][language]["mean_penalty_percent"]) < 1e-12 and abs(sd-saved_matched["per_language"][language]["sample_stddev_percentage_points"]) < 1e-12
    with (RESULTS / "master_results_table.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["evidence_tier", "experiment", "vocab_size", "replicate", "language", "condition", "tokens", "bytes_per_token", "roundtrip_exact", "note"]); writer.writerows(rows)
    write_figures(sweep, values, summary)
    write_budget_comparison(values, summary, matched_values, matched_summary)
    def condition_bytes(experiment, condition):
        return [experiment["replicates"][replicate]["models"][condition]["training_utf8_bytes"] for replicate in REPLICATES]
    def listed(numbers):
        text = [f"{number:,}" for number in numbers]
        return text[0] if len(text) == 1 else ", ".join(text[:-1]) + ", and " + text[-1]
    joint_bytes = condition_bytes(matched, "shared_multilingual")
    dedicated_bytes = [sum(condition_bytes(matched, f"{language}_separate")[index] for language in LANGUAGES) for index in range(len(REPLICATES))]
    repeated_joint_bytes = condition_bytes(repeated, "shared_multilingual")
    repeated_dedicated_bytes = [sum(condition_bytes(repeated, f"{language}_separate")[index] for language in LANGUAGES) for index in range(len(REPLICATES))]
    table = ["# Master Experiment Report", "", "Derived from saved JSON by `python src/consolidate_results.py`; no training was run and raw outputs were not modified.", "", "## Design", "", "Two complementary byte-BPE regimes use the same requested vocabulary and fixed held-out tests. The fixed-per-language regime trains each dedicated tokenizer on about 300 KB and the joint tokenizer on about 900 KB. The total-budget-matched regime trains dedicated and joint conditions from the same approximately 100 KB per-language samples; the joint tokenizer totals about 300 KB. Both are intrinsic token-count experiments, not downstream-quality evaluations.", "", "## Evidence tier 1: single-run vocabulary sweep", "", "One deterministic balanced run was saved at each size; these are observations, not replicate means.", "", "| Vocab | English | Hindi | Nepali |", "| --- | --- | --- | --- |"]
    for size in ("512", "1024", "2048"):
        vals=[]
        for language in LANGUAGES:
            dedicated, shared = metrics(sweep["models"][size], language); vals.append(f"{penalty(dedicated['tokens'], shared['tokens']):.2f}%")
        table.append(f"| {size} | {' | '.join(vals)} |")
    repeated_deltas = [joint - dedicated for joint, dedicated in zip(repeated_joint_bytes, repeated_dedicated_bytes)]
    repeated_delta_sentence = f"Each joint total is {repeated_deltas[0]} bytes larger than that sum because concatenation inserts a newline between the language samples." if len(set(repeated_deltas)) == 1 else f"Joint-minus-dedicated differences are {', '.join(str(delta) for delta in repeated_deltas)} bytes."
    table += ["", "`balanced_vocabulary_sharing_v1.json` repeats the same 1024 selection and is not an independent run.", "", "## Evidence tier 2: three-resample summary at vocabulary 1024", "", f"Each dedicated tokenizer trains on one language sample of about 300 KB. Joint training bytes are {listed(repeated_joint_bytes)}. The three dedicated models sum to {listed(repeated_dedicated_bytes)} bytes. {repeated_delta_sentence}", "", "| Language | Dedicated tokens (1/2/3) | Shared tokens (1/2/3) | Mean ± sample SD |", "| --- | --- | --- | --- |"]
    for language in LANGUAGES:
        dedicated = "/".join(str(x[0]) for x in replicate_details[language]); shared = "/".join(str(x[1]) for x in replicate_details[language]); mean, sd = summary[language]; table.append(f"| {language.title()} | {dedicated} | {shared} | {mean:.2f}% ± {sd:.2f} pp |")
    deltas = [joint - dedicated for joint, dedicated in zip(joint_bytes, dedicated_bytes)]
    delta_sentence = f"Each joint total is {deltas[0]} bytes larger than that sum because concatenation inserts a newline between the language samples." if len(set(deltas)) == 1 else f"Joint-minus-dedicated differences are {', '.join(str(delta) for delta in deltas)} bytes."
    table += ["", "## Evidence tier 3: total-budget-matched three-resample control at vocabulary 1024", "", f"Within each replicate, the dedicated and joint conditions use the same per-language source sample of approximately 100 KB. Joint training bytes are {listed(joint_bytes)}. The three dedicated models sum to {listed(dedicated_bytes)} bytes. {delta_sentence}", "", "| Language | Replicate | Dedicated tokens | Joint tokens | Penalty (%) | Exact reconstruction |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for language in LANGUAGES:
        for index, _replicate in enumerate(REPLICATES):
            dedicated_tokens, joint_tokens = matched_details[language][index]
            models = matched["replicates"][REPLICATES[index]]["models"]
            exact = models[f"{language}_separate"]["test"][language]["metrics"]["roundtrip_exact"] and models["shared_multilingual"]["test"][language]["metrics"]["roundtrip_exact"]
            table.append(f"| {language.title()} | {index + 1} | {dedicated_tokens} | {joint_tokens} | {matched_values[language][index]:.2f} | {'exact' if exact else 'not exact'} |")
    table += ["", "Exact reconstruction means both the dedicated tokenizer and the joint tokenizer encode and decode that language's held-out test without change.", "", "| Language | Mean ± sample SD |", "| --- | --- |"]
    for language in LANGUAGES:
        mean, sd = matched_summary[language]
        table.append(f"| {language.title()} | {mean:.2f}% ± {sd:.2f} pp |")
    table += ["", "## Claims and limitations", "", "- The shared 1024-token vocabulary produced more tokens than the dedicated vocabulary for every language in every saved resample in both budget regimes.", "- Mean penalties are lower under the approximately 300 KB joint budget than under the approximately 900 KB joint budget. Absolute test-token counts are higher for both dedicated and joint models in the matched regime, so the smaller percentage is not a reduction in joint-model tokens.", "- That cross-regime difference changes dedicated sample size, joint sample size, and document composition together. It does not isolate total training bytes as a cause.", "- For the approximately 300 KB-per-language resamples, document overlap has Jaccard ranges of English 0.14–0.23, Hindi 0.07–0.09, and Nepali 0.05–0.10. Sample SDs describe that resample spread, not confidence intervals or statistical significance. Overlap among the 100 KB subsets was not re-audited in this reporting pass.", "- Corpus checks cover hashes, budgets, duplicates, and normalization collisions, not upstream provenance or pre-split contamination.", "- Results are limited to this corpus, implementation, byte-BPE setup, budget allocation, and token-count metric. They do not establish novelty, downstream-quality effects, or broad generalization.", "", "All tabulated controlled runs reconstruct inputs exactly after encoding and decoding."]
    (RESULTS / "master_experiment_report.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    e, h, n = (summary[language] for language in LANGUAGES)
    matched_e, matched_h, matched_n = (matched_summary[language] for language in LANGUAGES)
    portfolio = f'''# Portfolio Results Summary: Fixed-Vocabulary Multilingual BPE

## Result

At vocabulary 1024, three saved approximately 300 KB-per-language resamples show shared-vocabulary token-count penalties of **{e[0]:.2f}% ± {e[1]:.2f} pp** (English), **{h[0]:.2f}% ± {h[1]:.2f} pp** (Hindi), and **{n[0]:.2f}% ± {n[1]:.2f} pp** (Nepali). In that regime each dedicated tokenizer trains on about 300 KB and the joint tokenizer on about 900 KB. Every saved resample produced more tokens with the shared vocabulary on its matched held-out test set.

The total-budget-matched control uses the same approximately 100 KB source sample for each corresponding dedicated and joint condition. Its joint tokenizer totals approximately 300 KB. Mean penalties remain positive: **{matched_e[0]:.2f}% ± {matched_e[1]:.2f} pp** (English), **{matched_h[0]:.2f}% ± {matched_h[1]:.2f} pp** (Hindi), and **{matched_n[0]:.2f}% ± {matched_n[1]:.2f} pp** (Nepali). These means are lower than in the approximately 900 KB joint regime, but absolute test-token counts are higher for both dedicated and joint models. The smaller percentage does not mean the joint tokenizer used fewer tokens, and the two regimes do not isolate training-budget size from sample size or composition.

The separate single-run 512/1024/2048 sweep also has positive penalties in all nine language-by-size observations. The sweep, the three-resample summary, and the budget comparison are separate evidence tiers.

This is a controlled empirical tokenization trade-off for this corpus and implementation, not evidence of statistical significance, algorithmic novelty, downstream-model impact, or universal multilingual behavior. Because resamples overlap, the reported standard deviations are descriptive only.

See `master_experiment_report.md`, `master_results_table.csv`, and `figures/` for the complete evidence.
'''
    (RESULTS / "portfolio_results_summary.md").write_text(portfolio, encoding="utf-8")
    print(f"Validated summaries; wrote {len(rows)} table rows, report, portfolio summary, and three SVG figures.")


if __name__ == "__main__":
    main()
