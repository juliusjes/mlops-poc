import argparse

import matplotlib.pyplot as plt
from ml_platform.shared.services.lakefs_service import LakeFSService


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uri",
        required=True,
        help="LakeFS data location containing NumPy arrays encoded as bytes",
    )

    parser.add_argument(
        "--sk",
        required=False,
        default="waferMap",
        help="Column name containing encoded NumPy arrays",
    )

    parser.add_argument(
        "--tk",
        required=False,
        default="failure_label",
        help="Column name containing labels",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    service = LakeFSService()
    print("Loading data...")
    df = service.lakeFS_parquet_to_df(
        args.uri,
        deser_col=[args.sk],
    )
    print("Done")

    sk = args.sk
    tk = args.tk

    i = 0

    fig, ax = plt.subplots()

    # Display the first wafer map
    im = ax.imshow(df[sk].iloc[i].squeeze())
    ax.set_title(f"Wafer map {i}, label={df[tk].iloc[i]}")

    plt.show(block=False)

    while True:
        user_input = input("Press Enter for next wafer map, or type 'q' to quit: ")

        if user_input.lower() == "q":
            break

        i += 1

        if i >= len(df):
            print("Reached the end of the dataset.")
            break

        im.set_data(df[sk].iloc[i].squeeze())
        ax.set_title(f"Wafer map {i}, label={df[tk].iloc[i]}")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    plt.close(fig)


if __name__ == "__main__":
    main()