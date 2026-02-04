import argparse
import sys
import bbob_2009_functions
import plot_functions

def main():
    parser = argparse.ArgumentParser(description="Run or Plot BBOB 2009 functions.")
    parser.add_argument("func_num", type=int, help="Function number (1-14)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--degree", type=int, default=2, help="Rosenbrock degree (default: 2)")
    parser.add_argument("--plot", action="store_true", help="Plot the function in 3D")
    parser.add_argument("--save", type=str, help="Save the plot to a file")
    parser.add_argument("coords", type=float, nargs="*", help="Input coordinates (space-separated, required if not plotting)")

    args = parser.parse_args()

    class_name = f"function_{args.func_num}"
    
    if not hasattr(bbob_2009_functions, class_name):
        print(f"Error: Function {args.func_num} not found in bbob_2009_functions.py")
        sys.exit(1)

    if args.plot:
        plot_functions.plot_bbob_function(args.func_num, seed=args.seed, save_path=args.save)
        return

    if not args.coords:
        parser.error("coords are required when not plotting")

    func_class = getattr(bbob_2009_functions, class_name)
    func_instance = func_class(random_seed=args.seed, rosenbrock_degree=args.degree)
    
    try:
        result = func_instance.rosenbrock_fitness(args.coords)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error executing function_{args.func_num}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
