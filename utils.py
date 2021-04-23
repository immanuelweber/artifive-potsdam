import os
import re


def ld_to_dl(lst):
    # list of dicts to dict of lists
    return {key: [dic[key] for dic in lst] for key in lst[0]}


def dl_to_ld(dct):
    # dict of lists to list of dicts
    return [dict(zip(dct, t)) for t in zip(*dct.values())]


def get_files(path, regex_filter=None):
    """gets files in path, can be filtered using a regular expression

    Arguments:
        path {Path} -- path to find files in

    Keyword Arguments:
        regex_filter {str} -- regular expression to filter files (default: {None})

    Returns:
        list[str] -- files
    """
    if regex_filter is not None:
        return [file for file in os.listdir(path) if re.search(regex_filter, str(file))]
    return os.listdir(path)
