import sys
import pandas as pd
import glob
import os
from datetime import datetime, timedelta
import numpy as np
import configparser
# script searches and fills missing data from smartwatch data
# run with:
# python fill_missing_data.py [month] [day]
# BEFORE USEING MAKE BACKUP!
# needs folder named DAY_NUMBER in data folder
config = configparser.ConfigParser()
config.read('settings.config')

args = sys.argv
MONTH_NAME = args[1]
DAY_NUMBER = args[2]


PATH = config.get('fitbit.creds', 'DataPath') + MONTH_NAME + "/" + DAY_NUMBER


def add_date(df):
    df["time"] = pd.to_datetime(df["time"], format='%H:%M:%S')
    return df


def check_time_skip(start_dateS, end_dateS, myDates):
    myDates = [datetime.strptime(d, "%H:%M:%S") for d in myDates]
    start_date = datetime.strptime(start_dateS, "%H:%M:%S")
    end_date = datetime.strptime(end_dateS, "%H:%M:%S")

    completeDates = [start_date + timedelta(seconds=x)
                     for x in range(0, (end_date-start_date).seconds + 1)]
    completeDates = [d.strftime("%H:%M:%S")
                     for d in completeDates]  # Convert date to string

    myDates = [d.strftime("%H:%M:%S") for d in myDates]

    # Creates a list with missing data
    missingDates = [d for d in completeDates if d not in myDates]
    return missingDates


def load_files():
    os.chdir(PATH)
    files = glob.glob("*")
    if str(DAY_NUMBER) in files:
        files.remove(DAY_NUMBER)  # this folder is backup folder
    files.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d_%H-%M"))

    data = []
    for file in files:
        test = pd.read_csv(file, header=None)
        date = datetime.strptime(file, "%Y-%m-%d_%H-%M")
        test.columns = ['time', 'hrm', 'x', 'y', 'z']
        data.append(test)
    df = pd.concat(data)
    df = df.reset_index(drop=True)
    return df, files


def get_missing_dates(df, print_b=False):
    startT = df["time"][0]
    endT = df["time"].iloc[-1]
    missing_dates = check_time_skip(startT, endT, df["time"])
    if print_b == True:
        print(missing_dates)
        print("Number of missing dates: ", len(missing_dates))
    return missing_dates


def fix_missing_dates(df, missing_dates, files):
    df = add_date(df)

    df2 = pd.DataFrame({"time": missing_dates})
    df2 = add_date(df2)
    df = pd.concat([df, df2])
    df = df.sort_values(by='time')
    df = df.reset_index(drop=True)
    df = df.fillna(method='ffill')
    df["time"] = df["time"].dt.strftime('%H:%M:%S')
    path = PATH + "/" + DAY_NUMBER + "/"
    spans = np.arange(0, len(files) * 300, 300)

    for file, i in zip(files, spans):
        data = df[i:i+300].to_csv(header=False,
                                  index=False, line_terminator="\n")
        completeName = os.path.join(path, file)
        with open(completeName, "w") as file1:
            file1.write(data)


if __name__ == "__main__":
    df, files = load_files()
    missing_dates = get_missing_dates(df, print_b=False)
    fix_missing_dates(df, missing_dates, files)
