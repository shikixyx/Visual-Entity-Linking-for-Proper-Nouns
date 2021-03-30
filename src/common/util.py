import sys

sys.path.append(".")

import os
import pickle
import joblib
import datetime

from common.config import *


def load_object(filename):
    """
    Load pickle or joblib file
    """

    global object_dir
    ret = []

    if "." in filename:
        filename = filename.split(".")[0]

    for format in ["joblib", "pickle"]:
        file = object_dir + filename + "." + format

        if os.path.isfile(file):
            with open(file, "rb") as f:
                if format == "pickle":
                    ret = pickle.load(f)
                elif format == "joblib":
                    ret = joblib.load(f)

    return ret


def save_object(obj, filename, format="joblib"):
    """
    Save object to picke or joblib file
    """
    global object_dir

    if filename[-6:] == "joblib":
        format = "joblib"
        file = filename
    elif filename[-6:] == "pickle":
        format = "pickle"
        file = filename
    else:
        file = object_dir + filename + "." + format

    with open(file, "wb") as f:
        if format == "pickle":
            pickle.dump(obj, f)
        elif format == "joblib":
            joblib.dump(obj, f)

    return


def print_time_with_str(str=""):
    """
    Pring current time and str
    """
    now = datetime.datetime.now()
    print(now, str)
