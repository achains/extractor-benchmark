import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def read_benchmark_data(input_dir: str):
    benchmark_values = {}
    for filename in os.listdir(input_dir):
        algorithm_name, _ = filename.rstrip(".txt").split("_")
        benchmark_values.update(
            # [:-1] for removing trailing delimiter
            {algorithm_name: np.genfromtxt(os.path.join(input_dir, filename), delimiter=",")[:-1]}
        )

    return benchmark_values


def fps_graph(input_dir: str, output_dir: str):
    benchmark_values = read_benchmark_data(input_dir)
    frame_list = np.arange(0, len(list(benchmark_values.values())[0]))
    for algorithm_name, mks_time in benchmark_values.items():
        plt.plot(frame_list, 1e6 / mks_time, label=algorithm_name)

    plt.xlabel("frame")
    plt.ylabel("FPS")
    plt.legend()
    plt.show()


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    fps_graph(input_dir, output_dir)


if __name__ == "__main__":
    main()
