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

<table>
  <tr>
    <td align="center">1. Sphere Function<br><img src="plots/function_1.png" width="400"></td>
    <td align="center">2. Ellipsoidal Function<br><img src="plots/function_2.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">3. Rastrigin Function<br><img src="plots/function_3.png" width="400"></td>
    <td align="center">4. Büche-Rastrigin Function<br><img src="plots/function_4.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">5. Linear Slope Function<br><img src="plots/function_5.png" width="400"></td>
    <td align="center">6. Attractive Sector Function<br><img src="plots/function_6.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">7. Step-ellipsoidal Function<br><img src="plots/function_7.png" width="400"></td>
    <td align="center">8. Rosenbrock Function, original<br><img src="plots/function_8.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">9. Rosenbrock Function, rotated<br><img src="plots/function_9.png" width="400"></td>
    <td align="center">10. Ellipsoidal Function (High Conditioning)<br><img src="plots/function_10.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">11. Discus Function<br><img src="plots/function_11.png" width="400"></td>
    <td align="center">12. Bent Cigar Function<br><img src="plots/function_12.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">13. Sharp Ridge Function<br><img src="plots/function_13.png" width="400"></td>
    <td align="center">14. Different Powers Function<br><img src="plots/function_14.png" width="400"></td>
  </tr>
</table>


```bash
uv run python src/main.py <func_num> --plot [--save <filename>]
```

Example:
```bash
uv run python src/main.py 1 --plot --save plot.png
```
