import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Interactively plot CSV columns against sample_index."
    )
    parser.add_argument(
        "filename",
        help="Path to the CSV file"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.filename)

    if "sample_index" not in df.columns:
        print("Error: CSV does not contain a 'sample_index' column.")
        return

    plot_columns = []

    for column in df.columns:
        if column == "sample_index":
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")

        if numeric.notna().any():
            plot_columns.append(column)

    if not plot_columns:
        print("No numeric columns available for plotting.")
        return

    while True:
        print("\nColumns available for plotting against 'sample_index':")
        print("-" * 60)

        for i, column in enumerate(plot_columns, start=1):
            print(f"{i:3}: {column}")

        print("-" * 60)
        choice = input("Enter column number to plot, or 'q' to quit: ").strip()

        if choice.lower() == "q":
            break

        try:
            index = int(choice)
        except ValueError:
            print("Please enter a column number or 'q'.")
            continue

        if index < 1 or index > len(plot_columns):
            print("Invalid column number.")
            continue

        column = plot_columns[index - 1]

        # Non-numeric values become NaN and are excluded from the plot.
        y = pd.to_numeric(df[column], errors="coerce")
        valid = y.notna()

        x = df.loc[valid, "sample_index"]
        y = y.loc[valid]

        print(f"Plotting: {column}")
        print(f"Number of points: {len(x)}")

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("sample_index")
        plt.ylabel(column)
        plt.title(column)
        plt.tight_layout()
        plt.show()

        # Execution continues here after the plot window is closed.


if __name__ == "__main__":
    main()