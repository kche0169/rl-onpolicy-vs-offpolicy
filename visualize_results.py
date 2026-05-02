import json
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import compare_algorithms, plot_returns


def load_all_results(results_dir="results"):
    """Load all results from results directory"""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.startswith("all_results_") and filename.endswith(".json"):
            env_name = filename.replace("all_results_", "").replace(".json", "")
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                results[env_name] = json.load(f)
    return results


def plot_env_comparison(env_name, results, save_dir="results"):
    """Plot algorithm comparison for a single environment"""
    if env_name not in results:
        print(f"No results found for {env_name}")
        return

    env_results = results[env_name]

    # Compute mean returns across seeds
    returns_dict = {}
    for algo, algo_results in env_results.items():
        # Find the minimum episode length
        min_length = min(len(r["returns"]) for r in algo_results)
        trimmed_returns = [r["returns"][:min_length] for r in algo_results]
        mean_returns = np.mean(trimmed_returns, axis=0).tolist()
        returns_dict[algo] = mean_returns

    save_path = os.path.join(save_dir, f"comparison_{env_name}.png")
    compare_algorithms(returns_dict, window=50, save_path=save_path)
    print(f"Comparison plot saved to {save_path}")


def print_summary(env_name, results):
    """Print summary table for an environment"""
    if env_name not in results:
        print(f"No results found for {env_name}")
        return

    env_results = results[env_name]

    print(f"\n{'='*60}")
    print(f"Summary: {env_name}")
    print(f"{'='*60}")
    print(f"{'Algorithm':<15} {'Avg Return':>12} {'Std Return':>12} {'Avg Time (s)':>15}")
    print(f"{'-'*60}")

    for algo, algo_results in env_results.items():
        avg_returns = [r["eval_avg"] for r in algo_results]
        std_returns = [r["eval_std"] for r in algo_results]
        times = [r["train_time_sec"] for r in algo_results]

        mean_avg = np.mean(avg_returns)
        mean_std = np.mean(std_returns)
        mean_time = np.mean(times)

        print(f"{algo.upper():<15} {mean_avg:>12.2f} {mean_std:>12.2f} {mean_time:>15.1f}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize RL experiment results")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--env", type=str, default="all", help="Environment name (or 'all')")
    args = parser.parse_args()

    all_results = load_all_results(args.results_dir)

    if args.env == "all":
        for env_name in sorted(all_results.keys()):
            print_summary(env_name, all_results)
            plot_env_comparison(env_name, all_results, args.results_dir)
    else:
        print_summary(args.env, all_results)
        plot_env_comparison(args.env, all_results, args.results_dir)

    print("\nDone! Check the results directory for plots.")


if __name__ == "__main__":
    main()
