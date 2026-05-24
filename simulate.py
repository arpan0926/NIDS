"""
simulate.py
-----------
Simulates real-time network traffic classification.
Feeds test records through the trained model one-by-one,
mimicking a live intrusion detection system.

Usage:
    python simulate.py --n 50 --delay 0.1 --mode multiclass
"""

import time
import argparse
import numpy as np
import joblib
from preprocess import preprocess
from columns import LABEL_ENCODING

LABEL_NAMES = {v: k for k, v in LABEL_ENCODING.items()}
LABEL_NAMES_BINARY = {0: 'Normal', 1: 'Attack'}

# ANSI color codes for terminal output
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RESET  = '\033[0m'
BOLD   = '\033[1m'

THREAT_COLORS = {
    'normal': GREEN,
    'DoS':    RED,
    'Probe':  YELLOW,
    'R2L':    RED,
    'U2R':    RED,
    'Attack': RED,
    'Normal': GREEN,
}


def run_simulation(n: int = 30, delay: float = 0.2, mode: str = 'multiclass'):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  NIDS — Real-Time Traffic Monitor  |  mode={mode}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Load model and test data
    model = joblib.load('models/best_model.pkl')
    _, X_test, _, y_test, _ = preprocess(mode=mode, save_artifacts=False)

    label_names = LABEL_NAMES if mode == 'multiclass' else LABEL_NAMES_BINARY

    # Shuffle for variety
    indices = np.random.permutation(len(X_test))[:n]

    correct = 0
    alerts  = 0

    print(f"{'#':<5} {'True Label':<14} {'Predicted':<14} {'Status'}")
    print('-' * 55)

    for i, idx in enumerate(indices, 1):
        x      = X_test[idx].reshape(1, -1)
        true_l = label_names[y_test[idx]]
        pred_l = label_names[model.predict(x)[0]]
        match  = (true_l == pred_l)

        color  = THREAT_COLORS.get(pred_l, CYAN)
        status = f"{GREEN}✓ OK{RESET}" if match else f"{RED}✗ MISS{RESET}"
        threat = f"{RED}⚠ ALERT{RESET}" if pred_l not in ('Normal', 'normal') else ""

        print(f"{i:<5} {true_l:<14} {color}{pred_l:<14}{RESET} {status}  {threat}")

        if match:
            correct += 1
        if pred_l not in ('Normal', 'normal'):
            alerts += 1

        time.sleep(delay)

    print('-' * 55)
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Packets analyzed : {n}")
    print(f"  Correctly classified : {correct}/{n}  ({100*correct/n:.1f}%)")
    print(f"  Threats detected : {GREEN if alerts == 0 else RED}{alerts}{RESET}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate real-time NIDS')
    parser.add_argument('--n',     type=int,   default=30,          help='Number of packets to simulate')
    parser.add_argument('--delay', type=float, default=0.15,        help='Delay between packets (seconds)')
    parser.add_argument('--mode',  type=str,   default='multiclass', choices=['binary', 'multiclass'])
    args = parser.parse_args()

    run_simulation(n=args.n, delay=args.delay, mode=args.mode)
