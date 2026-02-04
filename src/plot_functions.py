import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import bbob_2009_functions
import sys

def plot_bbob_function(func_num, seed=42, range_min=-5, range_max=5, resolution=100, save_path=None):
    """
    Plots a BBOB 2009 function in 3D with a color gradient from red (minima) to blue (farthest).
    
    Args:
        func_num (int): The function number (1-24, though only 1-14 seem implemented).
        seed (int): Random seed for function generation.
        range_min (float): Minimum value for x and y axes.
        range_max (float): Maximum value for x and y axes.
        resolution (int): Number of points along each axis.
        save_path (str): If provided, saves the plot to this path. 
                        If it's just a filename, it will be saved in the 'plots' directory.
    """
    class_name = f"function_{func_num}"
    if not hasattr(bbob_2009_functions, class_name):
        print(f"Error: Function {func_num} not found.")
        return

    func_class = getattr(bbob_2009_functions, class_name)
    # BBOB functions in 2D for plotting
    func_instance = func_class(random_seed=seed, rosenbrock_degree=2)

    # Create grid
    x = np.linspace(range_min, range_max, resolution)
    y = np.linspace(range_min, range_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute Z values
    Z = np.zeros(X.shape)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func_instance.rosenbrock_fitness([X[i, j], Y[i, j]])

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Red is closest to global minima, blue is farthest.
    # The default 'jet' or 'coolwarm' doesn't exactly match "red = low, blue = high".
    # 'jet_r' (reversed jet) or custom map. 
    # jet goes Blue-Cyan-Yellow-Red. jet_r goes Red-Yellow-Cyan-Blue.
    # So Red (min) to Blue (max).
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet_r, linewidth=0, antialiased=True)

    # Set view angle: slight angle
    ax.view_init(elev=20, azim=30)
    
    # "Zoom in" by adjusting the plot margins
    fig.tight_layout()

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if save_path:
        # If save_path is just a filename, prepend 'plots/'
        if not os.path.dirname(save_path):
            save_path = os.path.join("plots", save_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_functions.py <func_num> [seed] [save_path]")
    else:
        fn = int(sys.argv[1])
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        sp = sys.argv[3] if len(sys.argv) > 3 else None
        plot_bbob_function(fn, seed=sd, save_path=sp)
