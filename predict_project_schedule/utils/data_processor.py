import pandas as pd

from dateutil.parser import parse

def open_csv(path, header=0):

    data = pd.read_csv(path, header=header)
    return data

def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False