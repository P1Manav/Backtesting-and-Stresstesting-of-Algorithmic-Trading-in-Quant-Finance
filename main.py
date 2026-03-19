"""Entry point routing to backtesting or stress testing modules"""
import sys
from pathlib import Path
import subprocess

def main():
    """Route to backtesting or stress testing main"""
    print("\n" + "=" * 80)
    print("ALGORITHMIC TRADING RESEARCH PLATFORM")
    print("=" * 80)
    print("\n  1. Backtesting")
    print("     - Run baseline performance analysis\n")
    print("  2. Stress Testing")
    print("     - Evaluate model robustness\n")
    print("-" * 80)
    choice = input("Select mode (1 or 2): ").strip()
    if choice == '1':
        backtest_main = Path(__file__).parent / 'backtesting_tool' / 'main.py'
        subprocess.run([sys.executable, str(backtest_main)])
    elif choice == '2':
        stress_main = Path(__file__).parent / 'stress_testing' / 'main.py'
        subprocess.run([sys.executable, str(stress_main)])
    else:
        print("[ERROR] Invalid choice")
        sys.exit(1)

if __name__ == '__main__':
    main()

