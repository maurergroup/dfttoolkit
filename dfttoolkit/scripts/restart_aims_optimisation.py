import argparse
import os
import re
import shutil
import subprocess
import sys
from os.path import isdir, isfile, join

import numpy as np
from aimstools.GeometryFile import GeometryFile
from aimstools.Utilities import (
    findAimsErrorFile,
    findAimsJobFile,
    findAimsOutputFile,
    queryYesNo,
)


def removeHessian(geometry_file):
    g = GeometryFile(geometry_file)
    g.saveToFile(geometry_file)


def prepareRestart(
    calc_dir, auto_yes, keep_hessian, output_name="", error_name=""
):
    if len(error_name) == 0:
        error_name = findAimsErrorFile(calc_dir)
    if len(output_name) == 0:
        output_name = findAimsOutputFile(calc_dir)
    error_file = join(calc_dir, error_name)
    output_file = join(calc_dir, output_name)
    geometry_file = join(calc_dir, "geometry.in")
    next_step_file = join(calc_dir, "geometry.in.next_step")
    control_file = join(calc_dir, "control.in")
    hessian_file = join(calc_dir, "hessian.aims")
    assassin_log_file = join(calc_dir, "assassin.log")

    if not isfile(next_step_file):
        if auto_yes or queryYesNo(
            "No geometry.in.next_step file was found.\n"
            'Do you like to start from the geometry.in file ("yes"),\n'
            'or do you want to abort ("no")?' + directory,
            default="no",
        ):
            # NOTE: this will skip the rest of prepareRestart and continue with submitRestart!
            return
        # this will abort the whole restartGeomOpt script
        exit()

    restart_dirs = [
        d for d in os.listdir(calc_dir) if re.match("restart[0-9]*", d)
    ]
    n_restart = len(restart_dirs)
    backup_dir = join(calc_dir, f"restart{n_restart:02d}")
    if isdir(backup_dir):
        raise RuntimeError("Backup directory already exists")
    os.makedirs(backup_dir)

    # read out initial moment from original geometry
    init_moment_old = GeometryFile(geometry_file).initial_moment

    # backup original input/output files
    shutil.copy(control_file, backup_dir)
    shutil.copy(geometry_file, backup_dir)
    shutil.copy(next_step_file, backup_dir)
    if isfile(output_file):
        shutil.copy(output_file, backup_dir)
    if isfile(error_file):
        shutil.copy(error_file, backup_dir)
    if isfile(
        hessian_file
    ):  # hessian_file only exists for geometry optimizations
        shutil.copy(hessian_file, backup_dir)
        if not keep_hessian:
            os.remove(hessian_file)

    if isfile(assassin_log_file):  # assassin log file may not exist
        shutil.move(assassin_log_file, backup_dir)

    # additional if-condition to assure removal only happens
    # after everything is sucessfully copied
    if isfile(output_file):
        os.remove(output_file)
    if isfile(error_file):
        os.remove(error_file)
    shutil.move(next_step_file, geometry_file)

    # transfer initial moment, if present
    if np.any(init_moment_old):
        geom = GeometryFile(geometry_file)
        geom.initial_moment = init_moment_old
        geom.saveToFile(geometry_file)


def submitRestart(calc_dir, jobfile_name=""):
    if len(jobfile_name) == 0:
        jobfile_name = findAimsJobFile(calc_dir)
    current_dir = os.getcwd()
    os.chdir(calc_dir)
    subprocess.call(["sbatch", jobfile_name])
    os.chdir(current_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-y",
        "--yes",
        dest="auto_yes",
        action="store_true",
        default=False,
        help="Do not ask for confirmation before restarting",
    )
    parser.add_argument(
        "--dry",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Do not submit the job to the queue",
    )
    parser.add_argument(
        "--keep-hessian",
        "-k",
        dest="keep_hessian",
        action="store_true",
        default=False,
        help="Do not remove the (estimated) hessian matrix from the geometry file",
    )
    parser.add_argument(
        "directory", default=".", nargs="?", help="Directory of calculation"
    )
    args = parser.parse_args(sys.argv)

    directory = os.getcwd()
    if args.auto_yes or queryYesNo(
        "Restart geometry optimization in: " + directory, default="no"
    ):
        prepareRestart(
            directory, auto_yes=args.auto_yes, keep_hessian=args.keep_hessian
        )

        # tldr; '--keep-hessian' does not work for aims version before end of 2018, but newer version should work
        # the following code is outdated:
        # new versions of aims (ca. 2019) use an external hessian.aims file
        # instead of storing the hessian in the geometry file.
        # Furthermore, the old hack (load as GeometryFile and save again to strip off the Hessian)
        # doesn't work anymore as the GeometryFile can now handle the (old in-line) Hessian, so it won't be removed.
        if not args.keep_hessian:
            removeHessian(join(directory, "geometry.in"))

        if not args.dry_run:
            submitRestart(directory)
