"""Example: solve the built-in 5-bus case."""

from fdls5bus import case5, solve_fdls


def main() -> None:
    result = solve_fdls(case5(), tolerance=1e-8)
    print(f"Converged: {result.converged}, iterations: {result.iterations}")
    for idx, (vm, va) in enumerate(zip(result.vm, result.va_deg), start=1):
        print(f"Bus {idx}: Vm={vm:.6f} pu, Va={va:.6f} deg")


if __name__ == "__main__":
    main()
