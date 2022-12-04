"""
Based on the information provided at:
https://en.wikipedia.org/wiki/Mandelbrot_set
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def non_negative_int_input(value):
    """
    A function for checking if an input value is a non-negative integer.

    :param value: the input value.
    :return: the non-negative integer value if this holds otherwise raises an exception.
    """

    try:
        # try to convert input value to integer
        value = int(value)

        # if conversion is successful check if the integer is non-negative
        if value < 0:
            # raise an exception if the integer is not a non-negative integer
            raise argparse.ArgumentTypeError(f"{value} is not a non-negative integer")
    except ValueError:

        # if conversion to integer fails then the input is not an integer
        raise argparse.ArgumentTypeError(f"{value} is not an integer.")

    # return the non-negative integer value if every process is successfully completed
    return value


def check_if_complex(value):
    """
    A function for checking if an input value is a complex number.

    :param value: the input value.
    :return: the complex number if this holds otherwise raises an exception.
    """

    try:
        # try to convert input value to a complex number
        value = complex(value)

    except ValueError:

        # if conversion to complex number fails then the input is not a complex number
        raise argparse.ArgumentTypeError(f"{value} is not a complex number.")

    # return the complex number if every process is successfully completed
    return value


def cmapping(z: complex, c: complex, deg: float) -> complex:
    """
    A function for computing the value of a mapping with the form  res = (z^deg) + c.

    :param z: the z term.
    :param c: the c term.
    :param deg: the degree of z.
    :return: the result of (z^deg) + c.
    """
    return (z ** deg) + c


def check_if_in_set(z0: complex, c: complex, deg: float, num_iter: int = 20, boundary: float = 2000) -> tuple:
    """
    A function for checking if a value of c is in the set for a given degree and z_{0} while concurrently finds the
    number of iterations for divergence or convergence.

    :param z0: the starting value of z_{n}.
    :param c: the c term.
    :param deg: the degree of z_{n}.
    :param num_iter: the number of iteration to check for divergence.
    :param boundary: the divergence boundary.
    :return: a tuple of the participation result of c (1 in set, 0 otherwise) and the number of iterations needed to diverge or converge.
    """

    zn = z0
    for iteration in range(num_iter):

        # compute the value for the next iteration
        z_np1 = cmapping(zn, c, deg)

        # if diverge return 0 (not in the set) and the number of iterations required to diverge
        if abs(z_np1) > boundary:
            return 0, iteration

        zn = z_np1

    # if the sequence converges return 1 (member of the set) and the number of iterations
    return 1, num_iter - 1


def main(z0: int = 0,
         deg: float = 2,
         c_a_bounds: tuple = (-2, 1),
         c_b_bounds: tuple = (-1.5, 1.5),
         n_points: int = 500,
         num_iter: int = 20,
         boundary: int = 2000,
         save_directory: str = None,
         format: str = "jpg"):
    """
    The main function for visualizing the Mandelbrot and related sets of the form (z^deg) + c

    :param z0: the starting value of z_{n}.
    :param deg: the degree of z_{n}.
    :param c_a_bounds: the min and max values for the real part of c terms to be examined as a tuple.
    :param c_b_bounds: the min and max values for the imaginary part of c terms to be examined as a tuple.
    :param n_points: the number of points to obtain between the specified min max intervals
    :param num_iter: the number of iterations to check for divergence.
    :param boundary: the convergence boundary.
    :param save_directory: the target directory to save the generated figures (creates it, if does not exist).
    :param format: the format to save the generated figures (choices: png, jpg, pdf).
    :return: None
    """

    # create a grid of points (c values)
    amin, amax = c_a_bounds
    bmin, bmax = c_b_bounds

    alphas = np.linspace(amin, amax, n_points)
    betas = np.linspace(bmin, bmax, n_points)
    alphas_grid, betas_grid = np.meshgrid(alphas, betas)

    # concatenate depth wise
    grid_of_points = np.dstack((alphas_grid, betas_grid))

    # reshape to a set of 2d points
    points = grid_of_points.reshape((-1, 2))

    # check if each c value is in set and get the number of iterations for convergence or divergence
    mandelbrot_set = [list(check_if_in_set(z0, complex(*point), deg, num_iter, boundary)) for point in points]

    # convert to numpy array and get the results for participation in set
    # and number of iterations for convergence or divergence
    mandelbrot_set_part_iter = np.array(mandelbrot_set)
    mandelbrot_set = mandelbrot_set_part_iter[:, 0]
    mandelbrot_set_iter = mandelbrot_set_part_iter[:, 1]

    # convert the results back to 2d form
    mandelbrot_set_structured = mandelbrot_set.reshape(grid_of_points.shape[:2])
    mandelbrot_set_iter_struc = mandelbrot_set_iter.reshape(grid_of_points.shape[:2])

    # set up the string for the figure title
    if (z0 == 0) and (deg == 2):
        title_str = f"The Mandelbrot Set $z_0={z0}, deg={deg}$"
    else:
        title_str = f"Set for $z_0={z0}, deg={deg}$"

    # visualize the set (specifically the c terms that are in and not in the set)
    fig = plt.figure(figsize=(12, 8))
    plt.contourf(alphas, betas, mandelbrot_set_structured, cmap="gray")
    plt.title(title_str)

    # save the figure to the target directory if specified
    if save_directory is not None:
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        plt.savefig(os.path.join(save_directory, f"produced_set.{format}"), dpi=300,
                    bbox_inches='tight')

    plt.show()

    # visualize the set (specifically number of iterations to diverge or converge)
    fig = plt.figure(figsize=(12, 8))
    plt.contourf(alphas, betas, mandelbrot_set_iter_struc, cmap="magma")
    plt.colorbar()
    plt.title(title_str + " - Iterations needed to diverge or converge")

    # save the figure to the target directory if specified
    if save_directory is not None:
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        plt.savefig(
            os.path.join(save_directory, f"produced_set_iterations_needed_diverge.{format}"),
            dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # create and parse arguments
    parser = argparse.ArgumentParser(description="A python script for visualizing Mandelbrot and related sets of\
    the form (z^deg) + c.")

    parser.add_argument("--z0", type=check_if_complex,
                        action="store", required=False,
                        default=0, help="The starting value of z_{n}.")
    parser.add_argument("--c_a_minmax", type=float,
                        action="store", nargs=2,
                        metavar=("C_AMIN", "C_AMAX"), required=False,
                        default=[-2, 1], help="The min and max values for the real part of c terms to be examined.")
    parser.add_argument("--c_b_minmax", type=float,
                        action="store", nargs=2,
                        metavar=("C_BMIN", "C_BMAX"), required=False,
                        default=[-1.5, 1.5], help="The min and max values for the imaginary part of c \
                        terms to be examined.")
    parser.add_argument("--n_points", type=non_negative_int_input,
                        action="store", required=False,
                        default=500, help="The number of points to obtain between the specified min max intervals.")
    parser.add_argument("--degree", type=float,
                        action="store", required=False,
                        default=2, help="The degree of z_{n}.")
    parser.add_argument("--nIter", type=non_negative_int_input,
                        action="store", required=False,
                        default=100, help="The number of iterations to check for divergence.")
    parser.add_argument("--boundary", type=non_negative_int_input,
                        action="store", required=False,
                        default=2000, help="The divergence boundary.")
    parser.add_argument("--savePath", type=str, default=None,
                        required=False,
                        help="The target directory to save the generated figures (if it does not exist,\
                         it will be created).")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "png", "pdf"],
                        required=False, help="The format to save the generated figures (choices: png, jpg, pdf).")
    args = parser.parse_args()

    # run the main function for the provided and default values
    main(z0=args.z0,
         deg=args.degree,
         c_a_bounds=args.c_a_minmax,
         c_b_bounds=args.c_b_minmax,
         n_points=args.n_points,
         num_iter=args.nIter,
         boundary=args.boundary,
         save_directory=args.savePath,
         format=args.format)
