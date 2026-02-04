# bbob_2009
Functions for the BBOB 2009 optimization challenge (https://coco.gforge.inria.fr/). 

In the python file, you will find some of the noiseless benchmark functions used in the bbob 2009 optimization challenge.

### Usage
You can run any function from the command line using the `run_function.sh` script:

```bash
./run_function.sh <func_num> <coords> [--seed <seed>] [--degree <degree>]
```

Example:
```bash
./run_function.sh 1 1.0 2.0 3.0
./run_function.sh 13 --seed 100 --degree 3 0.5 -0.5 1.0
```

### Plotting
You can also plot any of the implemented functions in 3D (figures are saved in the `plots` folder).
Currently, the following functions (1-14) are available for plotting in 2D:

1. Sphere Function
   ![Sphere Function](plots/function_1.png)
2. Ellipsoidal Function
   ![Ellipsoidal Function](plots/function_2.png)
3. Rastrigin Function
   ![Rastrigin Function](plots/function_3.png)
4. Büche-Rastrigin Function
   ![Büche-Rastrigin Function](plots/function_4.png)
5. Linear Slope Function
   ![Linear Slope Function](plots/function_5.png)
6. Attractive Sector Function
   ![Attractive Sector Function](plots/function_6.png)
7. Step-ellipsoidal Function
   ![Step-ellipsoidal Function](plots/function_7.png)
8. Rosenbrock Function, original
   ![Rosenbrock Function, original](plots/function_8.png)
9. Rosenbrock Function, rotated
   ![Rosenbrock Function, rotated](plots/function_9.png)
10. Ellipsoidal Function (High Conditioning)
    ![Ellipsoidal Function (High Conditioning)](plots/function_10.png)
11. Discus Function
    ![Discus Function](plots/function_11.png)
12. Bent Cigar Function
    ![Bent Cigar Function](plots/function_12.png)
13. Sharp Ridge Function
    ![Sharp Ridge Function](plots/function_13.png)
14. Different Powers Function
    ![Different Powers Function](plots/function_14.png)

```bash
uv run python src/main.py <func_num> --plot [--save <filename>]
```

Example:
```bash
uv run python src/main.py 1 --plot --save plot.png
```
