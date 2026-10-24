"""Run explicit stages of the source experiment from the project root."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import peak_forecasting as model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["historical", "inspect", "select", "leak-check", "combos", "final"])
    parser.add_argument("--data", type=Path, default=model.IN_XLSX)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/cli")
    parser.add_argument("--frozen", action="store_true", help="Confirm model and features were fixed using selection data.")
    args = parser.parse_args()

    if args.mode == "historical":
        table = model.pd.read_csv(PROJECT_ROOT / "results/historical_metrics.csv")
        print("Historical stored output; not rerun with this code.")
        print(table.to_string(index=False))
        return

    if args.mode == "final" and not args.frozen:
        parser.error("Final evaluation requires model/feature freezing. Use --frozen only after selection.")
    if not args.data.is_file():
        parser.error(f"Preprocessed Excel not found: {args.data}. See data/README.md.")

    df, usable, copy_days = model.prepare(model.add_candidates, path=args.data)
    if args.mode == "inspect":
        candidates_a, candidates_b = model.cand_lists(df.columns)
        print({
            "total_hours": len(df),
            "usable_hours": len(usable),
            "copied_days_excluded_from_training": len(copy_days),
            "candidates_A": len(candidates_a),
            "candidates_B": len(candidates_b),
            "final_features_A": len(model.FINAL_A),
            "final_features_B": len(model.FINAL_B),
        })
        return

    if args.mode == "leak-check":
        found = False
        for horizon, features in [("A", model.FINAL_A), ("B", model.FINAL_B)]:
            leaks = model.leak_test(model.add_candidates, features, horizon, path=args.data)
            print(horizon, leaks or "No future-power sensitivity at sampled cutoffs.")
            found |= bool(leaks)
        if found:
            raise SystemExit(1)
        return

    model.require_training_dependencies()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "select":
        for horizon, features in [("A", model.FEAT_A), ("B", model.FEAT_B)]:
            for name in ["LightGBM", "RandomForest"]:
                table = model.evaluate(usable, copy_days, {"baseline": features}, horizon, model=name)
                table.to_csv(args.output / f"selection_{horizon}_{name}.csv", encoding="utf-8-sig")
                print(horizon, name)
                print(table.round(3).to_string())
        return

    if args.mode == "combos":
        for horizon, sets, chosen, base in [
            ("A", model.A_COMBOS, model.PEAK_COMBO_MODEL_A, "A_기준"),
            ("B", model.B_COMBOS, model.PEAK_COMBO_MODEL_B, "B_기준"),
        ]:
            scores = model.evaluate(usable, copy_days, sets, horizon, model=chosen, seeds=model.SEEDS, period="select")
            ranked = model.rank_peak_combos(scores, base)
            ranked.to_csv(args.output / f"combos_{horizon}.csv", encoding="utf-8-sig")
            print(horizon, chosen)
            print(ranked.round(3).to_string())
        return

    table = model.final_score(
        usable, copy_days, model.FINAL_A, model.FINAL_B,
        frozen=True, out_dir=args.output,
        model_a=model.FINAL_MODEL_A, model_b=model.FINAL_MODEL_B,
    )
    print("Newly trained results; separate from historical_metrics.csv.")
    print(table.to_string())


if __name__ == "__main__":
    main()
