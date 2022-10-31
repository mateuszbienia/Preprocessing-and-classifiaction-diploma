import pandas as pd
import os
from datetime import datetime, timedelta
import glob
import numpy as np
from scipy.stats import kurtosis, skew
from feature_selector import FeatureSelector
import gather_keys_oauth2 as Oauth2
import fitbit
import configparser
config = configparser.ConfigParser()
config.read('settings.config')


CLIENT_ID = config.get('fitbit.creds', 'ClientID')
CLIENT_SECRET = config.get('fitbit.creds', 'ClientSecret')

path = config.get('PATH', 'DataPath')
SAVE_PATH = config.get('PATH', 'SavePath')


class DatasetCreator():
    def __init__(self, dates=None, path=None):

        self.dates = dates
        self.data_path = path

        self.sleep_auth2 = None

        self.raw_data = None
        self.processed_data = None

        self.chunk_size = 30

        self.labels = None

        self.predefined_skip_features = [
            'hrm_p10', 'accel_var', 'hrm_med', 'hrm_p90']

        self.epoch_times = []

        self.initialize_fitbit_api()

    def update_data(self, dates=None, path=None):
        self.dates = dates
        self.data_path = path
        self.raw_data = None
        self.processed_data = None
        self.epoch_times = []

    def initialize_fitbit_api(self):
        server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
        server.browser_authorize()
        ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
        REFRESH_TOKEN = str(
            server.fitbit.client.session.token['refresh_token'])
        auth2_client = fitbit.Fitbit(
            CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
        auth2_client.API_VERSION = 1.2

        self.sleep_auth2 = auth2_client

    def add_date_to_time(self, df, date):
        df["time"] = pd.to_datetime(
            df["time"], format='%H:%M:%S').dt.strftime('%H:%M:%S')
        date = date.strftime("%Y-%m-%d ")
        df['time'] = date + df['time'].astype(str)
        return df

    def generate_raw_dataset(self):
        df = []
        for d in self.dates:
            print("Processing... ", d.strftime('%Y-%m-%d'))
            df.append(self.generate_labaled_dataset(d))
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        self.raw_data = df

    def get_one_day(self, month, day):
        os.chdir(self.data_path + month + "/" + day)
        files = glob.glob("*")
        files.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d_%H-%M"))
        data = []
        for file in files:
            test = pd.read_csv(file, header=None)
            date = datetime.strptime(file, "%Y-%m-%d_%H-%M")
            test.columns = ['time', 'hrm', 'x', 'y', 'z']
            test = self.add_date_to_time(test, date)
            data.append(test)
        df = pd.concat(data)
        df = df.reset_index(drop=True)
        return df

    def get_sleep_data(self, date):
        sleep_data = self.sleep_auth2.sleep(date)
        sleep_dataS = sleep_data["sleep"][-1]["levels"]["data"]
        df = pd.DataFrame(sleep_dataS)
        # changes labels ["wake", "light", "deep", "rem"] into [0, 1, 2, 3]
        df["level"], self.labels = pd.factorize(df["level"])
        return df

    def get_sleep_short_data(self, date):
        sleep_data = self.sleep_auth2.sleep(date)
        sleep_dataS = sleep_data["sleep"][-1]["levels"]["shortData"]
        df = pd.DataFrame(sleep_dataS)
        df["level"], _ = pd.factorize(df["level"])
        return df

    def remove_miliseconds(self, df):
        date = []
        for row in df["dateTime"]:
            temp = datetime.strptime(row[0:-4], "%Y-%m-%dT%H:%M:%S")
            date.append(temp)
        return date

    def get_last_time_stage(self, date, seconds):
        return date + timedelta(seconds=int(seconds))

    def datetime_range(self, start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def make_labels(self, times, levels):
        labels = []
        for time, level in zip(times, levels):
            for _ in range(time):
                labels.append(level)
        return labels

    def make_labels_short_data(self, time, level):
        labels = []
        for _ in range(time):
            labels.append(level)
        return labels

    def generate_labaled_dataset(self, date):
        sleep_df = self.get_sleep_data(date)
        dates = self.remove_miliseconds(sleep_df)
        dates.append(self.get_last_time_stage(
            dates[-1], sleep_df["seconds"][sleep_df.index[-1]]))  # adds last
        dts = [dt.strftime('%Y-%m-%d %H:%M:%S')
               for dt in self.datetime_range(dates[0], dates[-1], timedelta(seconds=1))]

        labels = self.make_labels(sleep_df["seconds"], sleep_df["level"])

        sleep_df = pd.DataFrame({"time": dts, "label": labels})

        short_data = self.get_sleep_short_data(date)
        dates = self.remove_miliseconds(short_data)
        short_dates = []
        for idx, d in enumerate(dates):
            short_dates.append(
                [d, d + timedelta(seconds=int(short_data["seconds"][short_data.index[idx]]))])
        new_df = []
        for idx, d in enumerate(short_dates):
            dts = [dt.strftime('%Y-%m-%d %H:%M:%S')
                   for dt in self.datetime_range(d[0], d[-1], timedelta(seconds=1))]
            labels = self.make_labels_short_data(
                short_data["seconds"][short_data.index[idx]], short_data["level"][short_data.index[idx]])
            new_df.append(pd.DataFrame({"time": dts, "label": labels}))

        df = pd.concat(new_df)
        df = df.reset_index(drop=True)
        df["label"] = df["label"].replace(["wake"], 0)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        sleep_df['time'] = pd.to_datetime(sleep_df['time'])
        sleep_df.set_index('time', inplace=True)
        sleep_df.update(df)
        sleep_df['time'] = sleep_df.index.strftime('%Y-%m-%d %H:%M:%S')
        sleep_df = sleep_df.reset_index(drop=True)
        cols = sleep_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        sleep_df = sleep_df[cols]

        # csv_temp = sleep_df.to_csv(index=False, line_terminator="\n")
        # completeName = os.path.join(SAVE_PATH, "test.csv")
        # with open(completeName, "w") as f:
        #     f.write(csv_temp)

        dataset = self.get_one_day(
            month=date.strftime("%B"), day=str(date.day))
        df = pd.merge(dataset, sleep_df, how="left", on=["time"])
        df['label'] = df['label'].fillna(4)
        df['label'] = df['label'].astype('int64')
        if "awake" not in self.labels:
            self.labels = np.append(self.labels, "awake")

        return df

    def save_raw_data(self, filename):
        filename += ".csv"
        if self.raw_data is not None:
            temp = self.raw_data.copy()
            temp["label"] = self.labels[temp["label"]]
            csv = temp.to_csv(index=False, line_terminator="\n")
            completeName = os.path.join(SAVE_PATH, filename)
            with open(completeName, "w") as f:
                f.write(csv)
                print("File " + filename + " saved.")
        else:
            print("No raw data.")

    def save_processed_data(self, filename):
        filename += ".csv"
        if self.processed_data is not None:
            temp = self.processed_data.copy()
            csv = temp.to_csv(line_terminator="\n")
            completeName = os.path.join(SAVE_PATH, filename)
            with open(completeName, "w") as f:
                f.write(csv)
                print("File " + filename + " saved.")
        else:
            print("No raw data.")

    def load_raw_data(self, filename):
        self.raw_data = pd.read_csv(filename)

    def load_processed_data(self, filename):
        self.processed_data = pd.read_csv(filename)

    def features_labels(self, df, scaler=False):
        hrm = df["hrm"]
        accel = df["accel"]
        features = np.transpose(np.array([hrm, accel]))
        labels = df["label"].values.tolist()
        return features, labels

    def cut_data_to_chunks(self):
        features, labels = [], []
        span = timedelta(seconds=self.chunk_size)
        epoch_starts = []
        i = 0
        while i < self.processed_data.shape[0]-self.chunk_size:
            start = datetime.strptime(
                self.processed_data["time"][i], "%Y-%m-%d %H:%M:%S")
            modulo = start.second % self.chunk_size
            if modulo == 0:
                end = datetime.strptime(
                    self.processed_data["time"][i+self.chunk_size], "%Y-%m-%d %H:%M:%S")
                if end - start == span:
                    f, l = self.features_labels(
                        self.processed_data[i:i+self.chunk_size])
                    epoch_starts.append(start)
                    features.append(f)
                    labels.append(l)
                    i += self.chunk_size
                    continue
            else:
                i += 30 - modulo - 1
            i += 1
        self.epoch_times = [*self.epoch_times, *epoch_starts]
        return features, labels

    def process_chunks(self, data, dtype):
        result = []
        for d in data:
            if dtype == "feat":
                hrm = d[:, 0]
                accel = d[:, 1]
                row = [*self.process_chunk(hrm), *self.process_chunk(accel)]
                result.append(row)
            if dtype == "label":

                result.append(self.labels[int(np.around(np.mean(d), 0))])
        return result

    def process_chunk(self, data):
        #["mean", "min", "max", "med", "var", "kurtosis", "skew", "p10", "p90"]
        mean_v = np.around(np.mean(data), 3)
        min_v = np.around(np.min(data), 3)
        max_v = np.around(np.max(data), 3)
        med_v = np.around(np.median(data), 3)
        var_v = np.around(np.var(data), 3)
        kur_v = np.around(kurtosis(data), 3)
        skw_v = np.around(skew(data), 3)
        p10_v = np.around(np.percentile(data, 10), 3)
        p90_v = np.around(np.percentile(data, 90), 3)
        return [mean_v, min_v, max_v, med_v, var_v, kur_v, skw_v, p10_v, p90_v]

    def change_data_to_chunks(self):
        print("Chunking...")
        self.processed_data = pd.DataFrame(
            {"time": self.raw_data["time"], "hrm": self.raw_data["hrm"], "label": self.raw_data["label"]})
        self.process_accelerometer_data()
        features, labels = self.cut_data_to_chunks()
        features = self.process_chunks(features, dtype="feat")
        labels = self.process_chunks(labels, dtype="label")
        title = ['hrm_mean', 'hrm_min', 'hrm_max', 'hrm_med', 'hrm_var', 'hrm_kurtosis', 'hrm_skew', 'hrm_p10', 'hrm_p90',
                 'accel_mean', 'accel_min', 'accel_max', 'accel_med', 'accel_var', 'accel_kurtosis', 'accel_skew', 'accel_p10', 'accel_p90']

        self.processed_data = pd.DataFrame(data=features, columns=title)
        self.processed_data["label"] = labels
        self.processed_data.index = self.epoch_times

    def process_accelerometer_data(self):
        accel_x = self.raw_data["x"]
        accel_y = self.raw_data["y"]
        accel_z = self.raw_data["z"]

        accel_sum = np.sqrt(np.array(accel_x)**2 +
                            np.array(accel_y)**2 + np.array(accel_z)**2)
        self.processed_data["accel"] = accel_sum

    def remove_unnecessary_features(self, auto=False):
        if auto:
            self.processed_data = self.processed_data.drop(
                columns=self.predefined_skip_features)
        else:
            fs = FeatureSelector(data=self.processed_data.drop(
                "label", axis=1), labels=self.processed_data["label"])
            fs.identify_missing(missing_threshold=0.6)
            fs.identify_collinear(correlation_threshold=0.98)

            fs.identify_zero_importance(task='classification',
                                        eval_metric='auc',
                                        n_iterations=10,
                                        early_stopping=False)

            fs.identify_low_importance(cumulative_importance=0.99)
            fs.identify_single_unique()
            # Remove the features from all methods (returns a df)
            labels = self.processed_data["label"]
            self.processed_data = fs.remove(methods='all')
            self.processed_data["label"] = labels


def gen_dates(year, month, days):
    dates = []
    for d in days:
        dates.append(datetime(year=year, month=month, day=d))
    return dates

# ---------------------------------------------------------#
# examples of use:

# ---------------------------------------------------------#
# example 1:

# jan_days = [2,3,4,6,7,8,9, 10]
# dec_days = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]
# dates = []
# dates = [*dates, *gen_dates(2020, 12, dec_days)]
# dates = [*dates, *gen_dates(2021, 1, jan_days)]

# dc = DatasetCreator(dates, path)

# dc.generate_raw_dataset()

# dc.save_raw_data("raw_data")
# # dc.load_raw_data("D:/FitBiT/python_api/Fitbit_API/datasets/raw_data.csv")

# dc.change_data_to_chunks()
# dc.save_processed_data("processed_data")
# # dc.load_processed_data("D:/FitBiT/python_api/Fitbit_API/datasets/processed_data.csv")
# dc.remove_unnecessary_features()
# dc.save_processed_data("processed_data_selected_features")

# ---------------------------------------------------------#
# example 2:

# day_options = [
#     datetime(year = 2021, month = 1, day = 15),
#     datetime(year = 2021, month = 1, day = 16),
#     datetime(year = 2021, month = 1, day = 19),
#     datetime(year = 2021, month = 1, day = 21) ]
# file_names = ["2021_01_15", "2021_01_16", "2021_01_19", "2021_01_21"]

# for day, file_name in zip(day_options, file_names):
#     dc = DatasetCreator([day], path)
#     dc.generate_raw_dataset()
#     dc.change_data_to_chunks()
#     dc.remove_unnecessary_features(auto=True)
#     dc.save_processed_data(file_name)

# ---------------------------------------------------------#
