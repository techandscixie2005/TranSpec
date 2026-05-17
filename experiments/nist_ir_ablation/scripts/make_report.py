#!/usr/bin/env python3
"""
Generate experiment report from aggregated NIST IR ablation results.

Reads summary_mean_std.csv and ablation_effects.json, writes report.md
and report.html.

Usage:
  python make_report.py \\
    --summary_dir runs/nist_ir_ablation/summary \\
    --output_md runs/nist_ir_ablation/summary/report.md \\
    --output_html runs/nist_ir_ablation/summary/report.html
"""

import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NIST IR ablation experiment report",
    )
    parser.add_argument(
        "--summary_dir", required=True,
        help="Summary directory containing aggregated results",
    )
    parser.add_argument(
        "--output_md", required=True,
        help="Path for report.md output",
    )
    parser.add_argument(
        "--output_html", required=True,
        help="Path for report.html output",
    )
    return parser.parse_args()


def load_csv(path):
    """Load a CSV file into a list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_manifest(summary_dir):
    """Try to load preprocess_manifest.json for dataset info."""
    processed_common = os.path.join(
        os.path.dirname(summary_dir), "processed", "common",
    )
    manifest_path = os.path.join(processed_common, "preprocess_manifest.json")
    if os.path.exists(manifest_path):
        return load_json(manifest_path)
    return None


def generate_report_md(summary_dir):
    """Generate Markdown report content."""
    summary_path = os.path.join(summary_dir, "summary_mean_std.csv")
    effects_path = os.path.join(summary_dir, "ablation_effects.json")
    missing_path = os.path.join(summary_dir, "missing_runs.json")
    all_runs_path = os.path.join(summary_dir, "all_runs.csv")

    lines = []

    # Title
    lines.append("# NIST IR Ablation Experiment Report")
    lines.append("")

    # Dataset info
    manifest = load_manifest(summary_dir)
    if manifest:
        input_file = manifest.get("input_file", "N/A")
        lines.append(f"**Dataset:** `{input_file}`")
        total_records = manifest.get("num_processed", "N/A")
        lines.append(f"**Total records:** {total_records}")
        split = manifest.get("split", {})
        if split:
            lines.append(
                f"**Split:** Train={split.get('train', 'N/A')}, "
                f"Valid={split.get('valid', 'N/A')}, "
                f"Test={split.get('test', 'N/A')}"
                f" (seed={split.get('split_seed', 'N/A')})"
            )
        lines.append("")

    lines.append(f"**Summary directory:** `{summary_dir}`")
    lines.append("")

    # Experiment matrix
    lines.append("## Experiment Matrix")
    lines.append("")
    lines.append("| Condition | Spectral Encoding | Tokenizer |")
    lines.append("|---|---|---|")
    lines.append("| E0 (atom_nope) | Intensity only (1 channel) | Atom-level |")
    lines.append("| E1 (atom_fourier) | Intensity + Fourier (65 channels) | Atom-level |")
    lines.append("| E2 (spe_nope) | Intensity only (1 channel) | SPE (BPE subword) |")
    lines.append("| E3 (spe_fourier) | Intensity + Fourier (65 channels) | SPE (BPE subword) |")
    lines.append("")

    # Main results table
    lines.append("## Main Results (mean ± std across seeds)")
    lines.append("")
    lines.append("| Model | Top-1 | Top-3 | Top-5 | Top-10 | Invalid Rate | Avg Candidates |")
    lines.append("|---|---|---|---|---|---|---|")

    if os.path.exists(summary_path):
        rows = load_csv(summary_path)
        for row in rows:
            model = row.get("model_id", "?")
            t1 = row.get("top1_mean", "")
            t1s = row.get("top1_std", "")
            t3 = row.get("top3_mean", "")
            t3s = row.get("top3_std", "")
            t5 = row.get("top5_mean", "")
            t5s = row.get("top5_std", "")
            t10 = row.get("top10_mean", "")
            t10s = row.get("top10_std", "")

            inv = row.get("invalid_decode_rate_mean", "")
            inv_s = row.get("invalid_decode_rate_std", "")
            avg_c = row.get("avg_num_candidates_mean", "")
            avg_c_s = row.get("avg_num_candidates_std", "")

            def fmt(v, s):
                if v == "" or v is None:
                    return "—"
                try:
                    v_f = float(v)
                    s_f = float(s) if s != "" else 0.0
                    return f"{v_f:.4f} ± {s_f:.4f}"
                except (ValueError, TypeError):
                    return "—"

            lines.append(
                f"| {model} | {fmt(t1, t1s)} | {fmt(t3, t3s)} | "
                f"{fmt(t5, t5s)} | {fmt(t10, t10s)} | "
                f"{fmt(inv, inv_s)} | {fmt(avg_c, avg_c_s)} |"
            )
    else:
        lines.append("| *No summary data found* | — | — | — | — | — | — |")
    lines.append("")

    # Ablation effects
    lines.append("## Ablation Effects")
    lines.append("")
    lines.append(
        "Difference in Top-k accuracy relative to E0 (atom_nope baseline). "
        "Positive values indicate improvement over baseline."
    )
    lines.append("")
    lines.append("| Effect | Top-1 | Top-3 | Top-5 | Top-10 |")
    lines.append("|---|---|---|---|---|")

    if os.path.exists(effects_path):
        effects = load_json(effects_path)
        effect_labels = {
            "delta_pe": "Fourier PE (E1 − E0)",
            "delta_spe": "SPE (E2 − E0)",
            "delta_both": "Both (E3 − E0)",
            "interaction": "Interaction",
        }
        for effect_key, label in effect_labels.items():
            row_str = f"| {label} |"
            for metric in ["top1", "top3", "top5", "top10"]:
                metric_effects = effects.get(metric, {})
                val = metric_effects.get(effect_key, None)
                if val is not None:
                    row_str += f" {val:+.4f} |"
                else:
                    row_str += " — |"
            lines.append(row_str)
    else:
        lines.append("| *No ablation effects found* | — | — | — | — |")
    lines.append("")

    # Missing runs
    missing_count = 0
    if os.path.exists(missing_path):
        missing_data = load_json(missing_path)
        missing_list = missing_data.get("missing", [])
        missing_count = len(missing_list)
        if missing_list:
            lines.append("## Missing Runs")
            lines.append("")
            lines.append(
                f"The following {len(missing_list)} run(s) were not found or "
                f"incomplete:"
            )
            lines.append("")
            for m in missing_list:
                model = m.get("model", "?")
                seed = m.get("seed", "")
                if seed:
                    lines.append(f"- `{model}/seed_{seed}`")
                else:
                    lines.append(f"- `{model}` (no runs found)")
            lines.append("")

    # Reproduction
    lines.append("## Reproduction Commands")
    lines.append("")
    lines.append("### Preprocessing")
    lines.append("```bash")
    if manifest:
        cfg = manifest.get("config_file", "configs/nist_ir_base.yaml")
        inp = manifest.get("input_file", "data/nist_ir/raw/IR_nist.jsonl")
        out = os.path.dirname(os.path.dirname(summary_dir))
        lines.append(
            f"python experiments/nist_ir_ablation/scripts/prepare_jsonl.py \\\n"
            f"  --input {inp} \\\n"
            f"  --output {out} \\\n"
            f"  --config experiments/nist_ir_ablation/configs/{os.path.basename(cfg)}"
        )
    else:
        lines.append("# Preprocessing command not available")
    lines.append("```")
    lines.append("")

    lines.append("### Sequential Run")
    lines.append("```bash")
    lines.append(
        "# Adjust paths as needed"
    )
    lines.append(
        "bash experiments/nist_ir_ablation/run_full_12.sh \\\n"
        "  --input data/nist_ir/raw/IR_nist.jsonl \\\n"
        "  --output runs/nist_ir_ablation"
    )
    lines.append("```")
    lines.append("")

    lines.append("### Slurm Array")
    lines.append("```bash")
    lines.append("# Step 1: Preprocess")
    lines.append(
        "python experiments/nist_ir_ablation/scripts/prepare_jsonl.py \\\n"
        "  --input data/nist_ir/raw/IR_nist.jsonl \\\n"
        "  --output runs/nist_ir_ablation \\\n"
        "  --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml"
    )
    lines.append("")
    lines.append("# Step 2: Submit array")
    lines.append("sbatch experiments/nist_ir_ablation/run_full_12.slurm")
    lines.append("")
    lines.append("# Step 3: Aggregate after completion")
    lines.append(
        "python experiments/nist_ir_ablation/scripts/aggregate_results.py \\\n"
        "  --run_dir runs/nist_ir_ablation/runs \\\n"
        "  --summary_dir runs/nist_ir_ablation/summary"
    )
    lines.append("")
    lines.append("# Step 4: Generate report")
    lines.append(
        "python experiments/nist_ir_ablation/scripts/make_report.py \\\n"
        "  --summary_dir runs/nist_ir_ablation/summary \\\n"
        "  --output_md runs/nist_ir_ablation/summary/report.md \\\n"
        "  --output_html runs/nist_ir_ablation/summary/report.html"
    )
    lines.append("```")
    lines.append("")

    # Smoke test
    lines.append("### Smoke Test")
    lines.append("```bash")
    lines.append(
        "bash experiments/nist_ir_ablation/run_smoke_200.sh \\\n"
        "  --input data/nist_ir/raw/IR_nist_200.jsonl \\\n"
        "  --output runs/nist_ir_ablation_smoke"
    )
    lines.append("```")
    lines.append("")

    # Preliminary conclusions
    lines.append("## Preliminary Conclusions")
    lines.append("")
    lines.append(
        "*Note: These are directional observations based on limited seeds. "
        "Statistical significance testing requires more replicates.*"
    )
    lines.append("")

    if os.path.exists(effects_path):
        effects = load_json(effects_path)
        t1_effects = effects.get("top1", {})

        pe_effect = t1_effects.get("delta_pe", None)
        spe_effect = t1_effects.get("delta_spe", None)
        both_effect = t1_effects.get("delta_both", None)

        if pe_effect is not None:
            pe_desc = "positive" if pe_effect > 0 else "negative" if pe_effect < 0 else "neutral"
            lines.append(f"- **Fourier PE:** {pe_desc} (Top-1 delta = {pe_effect:+.4f})")

        if spe_effect is not None:
            spe_desc = "positive" if spe_effect > 0 else "negative" if spe_effect < 0 else "neutral"
            lines.append(f"- **SPE:** {spe_desc} (Top-1 delta = {spe_effect:+.4f})")

        if both_effect is not None:
            both_desc = "best" if both_effect >= max(
                pe_effect or 0, spe_effect or 0, 0
            ) else "not best"
            lines.append(f"- **Combined (Fourier + SPE):** {both_desc} (Top-1 delta = {both_effect:+.4f})")

        interaction = t1_effects.get("interaction", None)
        if interaction is not None:
            lines.append(f"- **Interaction:** {interaction:+.4f}")
    else:
        lines.append("- No effects data available yet.")

    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `make_report.py`*")
    lines.append("")

    return "\n".join(lines)


def md_to_html(md_content):
    """Minimal Markdown to HTML conversion. Simple and functional."""
    import html as html_mod
    lines = md_content.split("\n")
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "<title>NIST IR Ablation Report</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; ",
        "         max-width: 960px; margin: 0 auto; padding: 2em; line-height: 1.6; color: #333; }",
        "  h1 { border-bottom: 2px solid #eee; padding-bottom: 0.3em; }",
        "  h2 { border-bottom: 1px solid #eee; padding-bottom: 0.2em; }",
        "  table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "  th { background-color: #f5f5f5; font-weight: bold; }",
        "  tr:nth-child(even) { background-color: #fafafa; }",
        "  code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }",
        "  pre { background-color: #f4f4f4; padding: 1em; border-radius: 4px; overflow-x: auto; }",
        "  pre code { background: none; padding: 0; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    in_table = False
    in_code = False
    code_buffer = []

    for line in lines:
        # Code block handling
        if line.startswith("```"):
            if in_code:
                code_text = html_mod.escape("\n".join(code_buffer))
                html_parts.append(f"<pre><code>{code_text}</code></pre>")
                code_buffer = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buffer.append(line)
            continue

        # Tables
        if line.startswith("|") and line.endswith("|") and "---" not in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not in_table:
                in_table = True
                html_parts.append("<table>")
            html_parts.append("<tr>" + "".join(f"<td>{cells[i]}</td>" for i in range(len(cells))) + "</tr>")
            continue
        else:
            if in_table:
                html_parts.append("</table>")
                in_table = False

        # Separator row (---|---|---)
        if "|---|---" in line:
            continue

        # Empty lines
        if not line.strip():
            html_parts.append("<br>")
            continue

        # Headers
        if line.startswith("# "):
            html_parts.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_parts.append(f"<h2>{line[2:]}</h2>")

        # Bold
        elif line.startswith("**") and line.endswith("**"):
            html_parts.append(f"<p><strong>{line.strip('*')}</strong></p>")

        # List items
        elif line.startswith("- "):
            html_parts.append(f"<li>{line[2:]}</li>")

        # Horizontal rule
        elif line.startswith("---"):
            html_parts.append("<hr>")

        # Regular paragraph
        elif line.strip():
            escaped = html_mod.escape(line)
            # Convert inline code
            escaped = escaped.replace("`", "")
            html_parts.append(f"<p>{escaped}</p>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def main():
    args = parse_args()
    summary_dir = os.path.abspath(args.summary_dir)

    print(f"Summary dir: {summary_dir}")
    print(f"Output MD: {args.output_md}")
    print(f"Output HTML: {args.output_html}")

    # Generate report
    md_content = generate_report_md(summary_dir)

    # Write markdown
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write(md_content)
    print(f"  Written: {args.output_md}")

    # Write HTML
    html_content = md_to_html(md_content)
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    with open(args.output_html, "w") as f:
        f.write(html_content)
    print(f"  Written: {args.output_html}")

    print("Report generation complete.")


if __name__ == "__main__":
    main()
