import sys

sys.path.append(".")

import os
import subprocess

from common.config import *


def set_to_execfile(exec_file, from_str, to_str):
    """
    set str to exec file
    PARTITHIN,JOBNAME,OUTFILE,PYTHON,SRC,ARGV1-4
    """

    ## 出力先のファイルが指定
    if from_str == "OUTFILE":
        dir = os.path.dirname(to_str)

        ## ディレクトリ を作る
        if not os.path.isdir(dir):
            os.makedirs(dir)

    to_str = to_str.replace("/", "\/")
    cmd = "sed -i -e 's/{}/{}/g' {}".format(from_str, to_str, exec_file)
    restore_cmd = "sed -i -e 's/{}/{}/g' {}".format(to_str, from_str, exec_file)

    subprocess.call(cmd, shell=True)

    return restore_cmd


def print_parameters_to_logdir(log_dir, parameters):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = log_dir + "/parameters.txt"

    print("write parameters in ", log_file)
    with open(log_file, mode="a") as f:
        print(parameters, file=f)

    return


def sbatch_execfile(exec_file):
    cmd = "sbatch {}".format(exec_file)
    print(cmd)
    subprocess.call(cmd, shell=True)
