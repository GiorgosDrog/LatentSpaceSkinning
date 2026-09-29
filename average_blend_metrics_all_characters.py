import os
import numpy as np

from average_blend_metrics import compute_per_method, METHODS, KEYS, COL_LABELS, _tex_escape, TABLES_DIR

CHARACTERS = ["michelle", "alpha", "amy", "monster", "mouse"]


def main():
    per_method_all = {name: {k: [] for k in KEYS} for name, _ in METHODS}
    per_character_counts = {name: 0 for name, _ in METHODS}

    for character in CHARACTERS:
        per_method, _, skipped, _ = compute_per_method(character)
        if skipped:
            print(f"[{character}] {len(skipped)} missing output(s) -- excluded from that character's own average, "
                  f"see average_blend_metrics.py {character} for details")
        for name, _ in METHODS:
            rows = per_method[name]
            if not rows:
                print(f"[{character}] {name}: NO data at all -- excluded from the all-character average")
                continue
            char_avg = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
            for k in KEYS:
                per_method_all[name][k].append(char_avg[k])
            per_character_counts[name] += 1

    header = f"{'method':<18}{'n_char':>7}" + "".join(f"{k:>12}" for k in KEYS)
    print(f"\nAll characters ({', '.join(CHARACTERS)}): mean of each character's own mean\n")
    print(header)
    print("-" * len(header))
    final_avg = {}
    for name, _ in METHODS:
        n = per_character_counts[name]
        if n == 0:
            continue
        avg = {k: np.mean(per_method_all[name][k]) for k in KEYS}
        final_avg[name] = avg
        print(f"{name:<18}{n:>7}" + "".join(f"{avg[k]:>12.4f}" for k in KEYS))

    lines = [r"\begin{tabular}{l" + "r" * len(KEYS) + "}", r"\toprule",
             "Method & " + " & ".join(COL_LABELS[k] for k in KEYS) + r" \\", r"\midrule"]
    for name, _ in METHODS:
        if name not in final_avg:
            continue
        vals = " & ".join(f"{final_avg[name][k]:.4f}" for k in KEYS)
        lines.append(f"{_tex_escape(name)} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, "tab_synthesis_all.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
