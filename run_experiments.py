import argparse
import csv

import numpy as np

from src.config import DATA_DIR, HORIZONS, OUTPUT_DIR
from src.data_processing import read_raw_or_provided
from src.features import build_windows
from src.reporting import write_markdown_report
from src.training import train_one
from src.visualization import save_svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    data, split_idx, source_note = read_raw_or_provided()
    DATA_DIR.mkdir(exist_ok=True)
    data.to_csv(DATA_DIR / "daily_power.csv", index=False, encoding="utf-8-sig")
    print(f"Daily rows: {len(data)}, split index: {split_idx}, source: {source_note}", flush=True)

    all_rows = []
    summary_rows = []
    first_feature_cols = None
    for horizon in HORIZONS:
        arrays = build_windows(data, split_idx, horizon)
        first_feature_cols = arrays[5]
        print(f"Horizon {horizon}: train={len(arrays[0])}, test={len(arrays[2])}, features={len(arrays[5])}", flush=True)

        for model_name in ("lstm", "transformer", "cnn_transformer"):
            metrics = []
            model_predictions = []
            for seed in args.seeds:
                result = train_one(model_name, horizon, seed, arrays, args.epochs, args.batch_size, args.lr)
                metrics.append((result["mse"], result["mae"]))
                model_predictions.append(result)
                all_rows.append(
                    {
                        "horizon": horizon,
                        "model": model_name,
                        "seed": seed,
                        "mse": result["mse"],
                        "mae": result["mae"],
                    }
                )
                print(f"{model_name:16s} horizon={horizon} seed={seed} mse={result['mse']:.4f} mae={result['mae']:.4f}", flush=True)

            mse_values = np.array([m[0] for m in metrics])
            mae_values = np.array([m[1] for m in metrics])
            summary = {
                "horizon": horizon,
                "model": model_name,
                "mse_mean": float(mse_values.mean()),
                "mse_std": float(mse_values.std(ddof=1)),
                "mae_mean": float(mae_values.mean()),
                "mae_std": float(mae_values.std(ddof=1)),
            }
            summary_rows.append(summary)

            chosen = model_predictions[int(np.argmin(mse_values))]
            save_svg(
                fig_dir / f"{model_name}_h{horizon}.svg",
                chosen["dates"][0],
                chosen["truth"][0],
                chosen["pred"][0],
                f"{model_name} horizon={horizon}",
            )

    with (OUTPUT_DIR / "metrics_by_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "model", "seed", "mse", "mae"])
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUTPUT_DIR / "metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "model", "mse_mean", "mse_std", "mae_mean", "mae_std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown_report(DATA_DIR.parent / "实验报告.md", source_note, first_feature_cols or [], summary_rows)
    print(f"Done. Results saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
