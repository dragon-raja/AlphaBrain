"""Build standalone RL-search figures without changing the presentation PDF."""

from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

from AlphaBrain.research.dsol.plotting.search_ppo import (
    style, training_figure, evaluation_figure, task_figure, architecture_figure,
)


def build(run, output, font=None):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    style(font)
    updates = {seed: [json.loads(line) for line in (run / f"seed-{seed}" / "updates.jsonl").read_text().splitlines()]
               for seed in (0, 1, 2)}
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "evaluation.jsonl").read_text().splitlines()]
    figures = {
        "architecture": architecture_figure(), "training": training_figure(updates),
        "evaluation": evaluation_figure(summary), "tasks": task_figure(rows),
    }
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=160)
        plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    args = parser.parse_args()
    print(build(args.run, args.output, args.font))


if __name__ == "__main__":
    main()
