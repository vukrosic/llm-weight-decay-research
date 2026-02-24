#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from pathlib import Path


def subsample_singular_values(sigma_full, target_len):
    if not isinstance(sigma_full, list):
        return sigma_full
    if len(sigma_full) <= target_len:
        return sigma_full
    if target_len == 50:
        top_10 = sigma_full[:10]
        bottom_10 = sigma_full[-10:]
        middle = sigma_full[10:-10]
        step_size = max(1, len(middle) // 30)
        middle_sampled = middle[::step_size][:30]
        return top_10 + middle_sampled + bottom_10
    return sigma_full[:target_len]


def infer_reference_schema(reference_path, sample_lines):
    allowed_keys = set()
    sv_lengths = Counter()
    lines = 0

    with open(reference_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            allowed_keys.update(obj.keys())
            if isinstance(obj.get("singular_values"), list):
                sv_lengths[len(obj["singular_values"])] += 1
            lines += 1
            if lines >= sample_lines:
                break

    if not allowed_keys:
        raise ValueError(f"No valid JSON records found in reference file: {reference_path}")
    if not sv_lengths:
        raise ValueError("Reference file has no usable 'singular_values' lists.")

    target_sv_len = sv_lengths.most_common(1)[0][0]
    return allowed_keys, target_sv_len


def prune_file(input_path, output_path, allowed_keys, target_sv_len):
    total = 0
    written = 0
    keys_removed = 0
    sv_trimmed = 0
    invalid = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for raw in f_in:
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                invalid += 1
                continue

            filtered = {k: v for k, v in obj.items() if k in allowed_keys}
            keys_removed += len(obj) - len(filtered)

            if "singular_values" in filtered:
                sv_before = filtered["singular_values"]
                sv_after = subsample_singular_values(sv_before, target_sv_len)
                if isinstance(sv_before, list) and isinstance(sv_after, list) and len(sv_after) < len(sv_before):
                    sv_trimmed += 1
                filtered["singular_values"] = sv_after

            f_out.write(json.dumps(filtered) + "\n")
            written += 1

    return {
        "total": total,
        "written": written,
        "invalid": invalid,
        "keys_removed": keys_removed,
        "sv_trimmed": sv_trimmed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prune manifold_stats JSONL to match current tracked schema."
    )
    parser.add_argument("--input", required=True, help="Input manifold_stats.jsonl path")
    parser.add_argument("--reference", required=True, help="Reference JSONL with desired schema")
    parser.add_argument("--output", help="Output path (required unless --in-place)")
    parser.add_argument("--in-place", action="store_true", help="Rewrite input file in place")
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix when using --in-place (set empty string to disable backup)",
    )
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=5000,
        help="How many lines to sample from reference when inferring schema",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    reference_path = Path(args.reference)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    old_size = input_path.stat().st_size

    if args.in_place:
        tmp_output = input_path.with_suffix(input_path.suffix + ".tmp")
        output_path = tmp_output
    else:
        if not args.output:
            raise ValueError("Either --output or --in-place must be provided.")
        output_path = Path(args.output)

    allowed_keys, target_sv_len = infer_reference_schema(reference_path, args.sample_lines)
    stats = prune_file(input_path, output_path, allowed_keys, target_sv_len)

    if args.in_place:
        if args.backup_suffix:
            backup = input_path.with_name(input_path.name + args.backup_suffix)
            if backup.exists():
                backup.unlink()
            os.replace(input_path, backup)
            print(f"Backup written: {backup}")
        os.replace(output_path, input_path)

    new_size = (input_path if args.in_place else output_path).stat().st_size

    print("Prune complete")
    print(f"Allowed keys ({len(allowed_keys)}): {sorted(allowed_keys)}")
    print(f"Target singular_values length: {target_sv_len}")
    print(f"Records: {stats['total']} read, {stats['written']} written, {stats['invalid']} invalid skipped")
    print(f"Records with singular_values trimmed: {stats['sv_trimmed']}")
    print(f"Total keys removed: {stats['keys_removed']}")
    print(f"Size: {old_size / (1024 * 1024):.2f} MB -> {new_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
