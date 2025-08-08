import csv
import os
import subprocess
import pandas as pd
from bayes_opt import BayesianOptimization

# Path where simulation_metrics.csv is expected
CSV_PATH = os.path.join(
    "path/to/cloudsim-master/modules/cloudsim-examples", "simulation_metrics.csv")

iteration_counter = [0]


def run_simulation(mips, pes, ram):
    """
    Run the CloudSim simulation with specified VM parameters.
    """

    print(
        f"[INFO] Running simulation with MIPS={mips:.2f}, PEs={pes:.0f}, RAM={ram:.0f}")

    # Set environment variables so Java can access them
    env = os.environ.copy()
    env["VM_MIPS"] = str(int(mips))
    env["VM_PES"] = str(int(pes))
    env["VM_RAM"] = str(int(ram))

    try:
        result = subprocess.run(
            ["mvnd", "exec:java",
                "-Dexec.mainClass=org.cloudbus.cloudsim.examples.CloudSimOptimizationRunner"],
            cwd=".", capture_output=True, text=True, env=env, timeout=300
        )
        if result.returncode != 0:
            print("[ERROR] Simulation process failed:\n", result.stderr)
            return -1e6
    except subprocess.TimeoutExpired:
        print("[ERROR] Simulation timed out.")
        return -1e6

    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print("[ERROR] simulation_metrics.csv not found.")
        return -1e6

    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            print("[ERROR] simulation_metrics.csv is empty.")
            return -1e6
    except Exception as e:
        print(f"[ERROR] Failed to parse metrics: {e}")
        return -1e6

    # Objective: minimize avg_waiting and makespan
    try:
        avg_waiting = df["waiting"].mean()
        makespan = df["finish"].max() - df["arrival"].min()
        print(
            f"[INFO] Avg waiting: {avg_waiting:.2f}s, Makespan: {makespan:.2f}s")

        # You can design your own target — here is one example:
        target = - (avg_waiting + 0.1 * makespan)

        iteration_counter[0] += 1
        append_to_log(iteration_counter[0], mips, pes, ram, target)
        return target
    except KeyError as e:
        print(f"[ERROR] Missing column in CSV: {e}")
        return -1e6


LOG_FILE = "optimization_log.csv"


def append_to_log(iteration, mips, pes, ram, target):
    import csv

    try:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["iteration", "mips", "pes", "ram", "target"])
            writer.writerow([iteration, mips, pes, ram, target])
    except PermissionError:
        print(
            f"[ERROR] Cannot write to {LOG_FILE}. Is it open in another program?")
        exit(1)


def optimize():
    pbounds = {
        "mips": (100, 500),
        "pes": (50, 200),
        "ram": (2048, 16384)
    }

    optimizer = BayesianOptimization(
        f=run_simulation,
        pbounds=pbounds,
        verbose=2,
        random_state=42
    )

    optimizer.maximize(
        init_points=3,
        n_iter=15
    )

    print("\n=== Best Configuration Found ===")
    print(optimizer.max)


if __name__ == "__main__":
    optimize()
