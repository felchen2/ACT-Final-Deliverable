from pathlib import Path

from safety_data_prep import build_safety_dataset
from safety_visuals import create_safety_heatmap, create_safety_line_chart


def main():
    input_csv = Path("sa_exit_MASTER_merged_all4 (1).csv")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    cleaned_path, summary_path = build_safety_dataset(input_csv=input_csv, output_dir=output_dir)

    create_safety_line_chart(
        summary_csv=summary_path,
        output_png=output_dir / "safety_trend_chart.png",
        min_n=20,
    )

    create_safety_heatmap(
        summary_csv=summary_path,
        output_png=output_dir / "safety_heatmap.png",
        min_n=20,
    )

    print("Created:")
    print(cleaned_path)
    print(summary_path)
    print(output_dir / "safety_trend_chart.png")
    print(output_dir / "safety_heatmap.png")


if __name__ == "__main__":
    main()
