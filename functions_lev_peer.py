### supporting functions to machine_learning_calibration_and_site_transfer_test.ipynb
### for outlier removal, data loading, reading raw data, 
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK,\
ExpSineSquared as ESS, RationalQuadratic as RQ, Matern as M, DotProduct
import scipy.stats
import scipy.optimize as spo
import scipy.stats as scs
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score, recall_score
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def read_raw(filename,start_experiment, end_experiment, colocated_devices):
    # Read in file
    
    df = pd.read_csv(filename,  #dtype={"pm10": np.float64, "pm2_5": np.float64}, 
                     na_values=['Infinity'],
                     parse_dates=['timestamp']
                    )
    # filter by date and device
    
    df = df[(df.timestamp >= start_experiment) & (df.timestamp <= end_experiment) & (df.id.apply(lambda x : x in colocated_devices))]
    
    return df

def load_kings(path, start_experiment, end_experiment):
    df_kings_vol = pd.read_csv(path)
    df_kings_vol['timestamp'] = pd.to_datetime(df_kings_vol.date)
    # co is in ppm
    if 'co' in df_kings_vol.columns:
        df_kings_vol.co = 1000 * df_kings_vol.co

    df_kings_vol = df_kings_vol[ (df_kings_vol.timestamp > start_experiment) & (df_kings_vol.timestamp < end_experiment)]
    df_kings_vol = df_kings_vol.set_index('timestamp')
    ds_kings_vol = xr.Dataset.from_dataframe(df_kings_vol)
    
    return ds_kings_vol

# def find_start_end_coloc(df_coloc_times, sensor_id, ):
#     start = df_coloc_times[df_coloc_times.id == sensor_id]['good period of colocation at CR7: Start'].values[0]
#     end = df_coloc_times[df_coloc_times.id == sensor_id]['good period of colocation at CR7: End'].values[0]
#     return start, end


def preprocess(df, start_experiment):
    reading_time = df['timestamp']

    # Get hour, date, dayofweek
    df['hr'] = reading_time.apply(lambda x: x.hour)
    df['date'] = reading_time.apply(lambda x: x.date())
    df['dayofweek'] = reading_time.apply(lambda x: x.dayofweek + 1)  # Monday=1, Sunday=7
    df['month'] = reading_time.apply(lambda x: x.month)
    df['week'] = reading_time.apply(lambda x: x.week)
    df['count'] = 1

    df['time_since_start'] = reading_time.apply(lambda x: x - start_experiment)
    since_start = df['time_since_start']
    df['days_since_start'] = since_start.apply(lambda x: x.days)

    df['weeks_since_start'] = df['days_since_start'].apply(lambda x: int(x / 7))
    df.drop('time_since_start', axis=1)

    df = df.sort_values(by='timestamp')
    return df

### MAD outlier removal

def remove_outliers_PM10(df, thresh=3.5):
    def mad_based_outlier(points, thresh=3.5):
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    limits_dict = {}

    df_numeric = df.select_dtypes(include=[np.number])
    # cols = set(df_numeric.columns)
    # cols = cols.difference(set(['time', 'lat', 'long', 'fix', 'speed', 'alt', 'head', 'rpm', 'id', 'ver',
    #                             'index', 'vol', 'rtys']))
    cols = ['pm2hum', 'pm225a', 'pm2par25', 'pm1par5', 'pm2tmp', 'pm125a', 'pm210a', 'pm2par5', 'pm2par3',\
                                'pm125c', 'pm225c', 'pm1par10', 'pm11a', 'pm1par3', 'pm21c', 'pm21a',\
                                'pm110a', 'pm11c', 'pm1par25', 'pm210c', 'pm1tmp', 'pm110c', 'pm2par10', 'pm1hum']
    print(cols)
    df_no_na = df.dropna(subset=cols)
    print(df_no_na.shape,'df_no_na.shape')
    id_list = df_no_na.id.unique()
    id_list.sort()
    for col in cols:
        for device_id in id_list:
#             print(df_no_na[df_no_na.id == device_id].shape)
            df_device = df_no_na[df_no_na.id == device_id]
            data_col_id = df_device[col].values
            outliers = mad_based_outlier(data_col_id, thresh=thresh)
#             print(outliers.shape)
            n_entries = len(data_col_id)
            perc_outliers = sum(outliers) / n_entries
            if perc_outliers > 0.05:
                print("Column: %s Device id %i Number of entries %i Share of outliers %f" % (
                    col, device_id, n_entries, perc_outliers))
#             print(df_device.index.shape,'index.shape')
#             print(outliers.shape,'outliers shape')
#             print(df_no_na.shape,'df_no_na.shape')
            df_no_na.loc[df_device.index, col + '_outlier'] = outliers
#     print(df_no_na.columns)
    df_clean = df_no_na.copy()
    for col in cols:
        outlier_col = col + '_outlier'
        df_clean = df_clean[~df_clean[outlier_col]]
    print("Total Number of entries %i Perc remaining: %f" % (len(df_no_na), len(df_clean) / len(df_no_na)))
    return df_clean[df.columns], limits_dict

def remove_outliers_NO2(df, thresh=3.5):
    def mad_based_outlier(points, thresh=3.5):
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    limits_dict = {}

    df_numeric = df.select_dtypes(include=[np.number])
    # cols = set(df_numeric.columns)
    # cols = cols.difference(set(['time', 'lat', 'long', 'fix', 'speed', 'alt', 'head', 'rpm', 'id', 'ver',
    #                             'index', 'vol', 'rtys']))
    cols = ['afewrk1', 'afeaux1',  'afewrk2', 'afeaux2', 'afewrk3', 'afeaux3', 'afept1k', 'isbwrk','isbaux', 'pm1hum']
    print(cols)
    df_no_na = df.dropna(subset=cols)
    print(df_no_na.shape,'df_no_na.shape')
    id_list = df_no_na.id.unique()
    id_list.sort()
    for col in cols:
        for device_id in id_list:
#             print(df_no_na[df_no_na.id == device_id].shape)
            df_device = df_no_na[df_no_na.id == device_id]
            data_col_id = df_device[col].values
            outliers = mad_based_outlier(data_col_id, thresh=thresh)
#             print(outliers.shape)
            n_entries = len(data_col_id)
            perc_outliers = sum(outliers) / n_entries
            if perc_outliers > 0.05:
                print("Column: %s Device id %i Number of entries %i Share of outliers %f" % (
                    col, device_id, n_entries, perc_outliers))
#             print(df_device.index.shape,'index.shape')
#             print(outliers.shape,'outliers shape')
#             print(df_no_na.shape,'df_no_na.shape')
            df_no_na.loc[df_device.index, col + '_outlier'] = outliers
#     print(df_no_na.columns)
    df_clean = df_no_na.copy()
    for col in cols:
        outlier_col = col + '_outlier'
        df_clean = df_clean[~df_clean[outlier_col]]
    print("Total Number of entries %i Perc remaining: %f" % (len(df_no_na), len(df_clean) / len(df_no_na)))
    return df_clean[df.columns], limits_dict

def remove_outliers_NO2_flexible(df, cols, thresh=3.5):
    def mad_based_outlier(points, thresh=3.5):
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    limits_dict = {}

    df_numeric = df.select_dtypes(include=[np.number])
    # cols = set(df_numeric.columns)
    # cols = cols.difference(set(['time', 'lat', 'long', 'fix', 'speed', 'alt', 'head', 'rpm', 'id', 'ver',
    #                             'index', 'vol', 'rtys']))
    # cols = ['afewrk1', 'afeaux1',  'afewrk2', 'afeaux2', 'afewrk3', 'afeaux3', 'afept1k', 'isbwrk','isbaux', 'pm1hum']
    print(cols)
    df_no_na = df.dropna(subset=cols)
    print(df_no_na.shape,'df_no_na.shape')
    id_list = df_no_na.id.unique()
    id_list.sort()
    for col in cols:
        for device_id in id_list:
#             print(df_no_na[df_no_na.id == device_id].shape)
            df_device = df_no_na[df_no_na.id == device_id]
            data_col_id = df_device[col].values
            outliers = mad_based_outlier(data_col_id, thresh=thresh)
#             print(outliers.shape)
            n_entries = len(data_col_id)
            perc_outliers = sum(outliers) / n_entries
            if perc_outliers > 0.05:
                print("Column: %s Device id %i Number of entries %i Share of outliers %f" % (
                    col, device_id, n_entries, perc_outliers))
#             print(df_device.index.shape,'index.shape')
#             print(outliers.shape,'outliers shape')
#             print(df_no_na.shape,'df_no_na.shape')
            df_no_na.loc[df_device.index, col + '_outlier'] = outliers
#     print(df_no_na.columns)
    df_clean = df_no_na.copy()
    for col in cols:
        outlier_col = col + '_outlier'
        df_clean = df_clean[~df_clean[outlier_col]]
    print("Total Number of entries %i Perc remaining: %f" % (len(df_no_na), len(df_clean) / len(df_no_na)))
    return df_clean[df.columns], limits_dict


def remove_outliers(df, thresh=3.5):
    def mad_based_outlier(points, thresh=3.5):
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    limits_dict = {}

    df_numeric = df.select_dtypes(include=[np.number])
    cols = set(df_numeric.columns)
    cols = cols.difference(set(['time', 'lat', 'long', 'fix', 'speed', 'alt', 'head', 'rpm', 'id', 'ver',
                                'index', 'vol', 'rtys']))
    # cols = ['afewrk1', 'afeaux1',  'afewrk2', 'afeaux2', 'afewrk3', 'afeaux3', 'afept1k', 'isbwrk','isbaux', 'pm1hum']
    print(cols)
    df_no_na = df.dropna(subset=cols)
    print(df_no_na.shape,'df_no_na.shape')
    id_list = df_no_na.id.unique()
    id_list.sort()
    for col in cols:
        for device_id in id_list:
#             print(df_no_na[df_no_na.id == device_id].shape)
            df_device = df_no_na[df_no_na.id == device_id]
            data_col_id = df_device[col].values
            outliers = mad_based_outlier(data_col_id, thresh=thresh)
#             print(outliers.shape)
            n_entries = len(data_col_id)
            perc_outliers = sum(outliers) / n_entries
            if perc_outliers > 0.05:
                print("Column: %s Device id %i Number of entries %i Share of outliers %f" % (
                    col, device_id, n_entries, perc_outliers))
#             print(df_device.index.shape,'index.shape')
#             print(outliers.shape,'outliers shape')
#             print(df_no_na.shape,'df_no_na.shape')
            df_no_na.loc[df_device.index, col + '_outlier'] = outliers
#     print(df_no_na.columns)
    df_clean = df_no_na.copy()
    for col in cols:
        outlier_col = col + '_outlier'
        df_clean = df_clean[~df_clean[outlier_col]]
    print("Total Number of entries %i Perc remaining: %f" % (len(df_no_na), len(df_clean) / len(df_no_na)))
    return df_clean[df.columns], limits_dict


def prepareData(data, test_size=0.2, drop_cols=[]):
    data = pd.DataFrame(data.copy())

    test_index = int(len(data) * (1 - test_size))

    data = data.dropna()
    data = data.reset_index(drop=True)

    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test


def shift(arr, num, fill_value=np.NaN):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def add_alerts_to_np_array(value_col, alert_threshold=35,):
    # weird numpy where this doesn't work for boolean arrays
    this_hour_over_alert_threshold = (value_col > alert_threshold).astype(float)
#     print(this_hour_over_alert_threshold)
    hour_minus_one_over_alert_threshold = shift(this_hour_over_alert_threshold, 1, fill_value=np.NaN)
#     print(shift(this_hour_over_alert_threshold, 1))
#     print(hour_minus_one_over_alert_threshold)
    hour_minus_two_over_alert_threshold = shift(this_hour_over_alert_threshold, 2, fill_value=np.NaN)
#     print(hour_minus_two_over_alert_threshold)

    alert_triggered = ((this_hour_over_alert_threshold + hour_minus_one_over_alert_threshold + hour_minus_two_over_alert_threshold) >= 2)
    #df['alert_triggered'].astype(int).plot()
    #print(sum(df.alert_triggered/len(df)))
    return alert_triggered