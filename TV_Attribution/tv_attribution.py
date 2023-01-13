import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

import warnings
warnings.filterwarnings('ignore')

from kats.consts import TimeSeriesData

from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType, TrendChangeParameters
from kats.detectors.robust_stat_detection import RobustStatDetector

from kats.detectors.trend_mk import MKDetector


from kats.models.metalearner.get_metadata import GetMetaData
from kats.detectors.outlier import OutlierDetector

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

class sample:

    def read_data(path,freq='60min'):
        data = pd.read_csv(path)
        df = data.copy()
        # Column names should be 'time' for dates and 'value' for values.
        df.columns = ['time','value']
        # Setting 'time' as index to filling missing minutes and filling them with '0' values.
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index)

        # Creating date range between first date and last date of dataframe. freq parameter should be 'min' or 'T'.
        idx = pd.date_range(df.index[0].date().strftime('%Y-%m-%d'),df.index[-1].date().strftime('%Y-%m-%d'),freq='min')
        # Adding missing dates and filling them with zeros.
        df = df.reindex(idx, fill_value=0)
        # Date should be column with 'time' column name. 
        df = df.reset_index(drop=False).rename(columns={'index':'time'})
        # Convert dataframe to kats.TimeSeriesData

        df = df.set_index('time').resample(freq).sum()

        # Convert dataframe to kats.TimeSeriesData
        ts = TimeSeriesData(df.reset_index())


        return ts


    def changepoint_detection(ts,interest_window=[],bocp_type=BOCPDModelType.NORMAL_KNOWN_MODEL,robust_p_value=0.01,comparison_window:int = -2):
        if len(ts)>20000:
            ts = ts[-5000:]
        detector1 = CUSUMDetector(ts)
        detector2 = BOCPDetector(ts)
        detector3 = RobustStatDetector(ts)

        change_points1 = detector1.detector()
        detector1.plot(change_points1)
        plt.xticks(rotation=45)
        plt.savefig('cusum_detector.png')
        if len(interest_window)==0:
            change_points2 = change_points1
            detector1.plot(change_points2)
            plt.xticks(rotation=45)
            plt.savefig('cusum_detector_interest_window.png')
        else:
            change_points2 = detector1.detector(interest_window=interest_window)
            detector1.plot(change_points2)
            plt.xticks(rotation=45)
            plt.savefig('cusum_detector_interest_window.png')
        change_points3 = detector2.detector(model=bocp_type)
        detector2.plot(change_points3)
        plt.xticks(rotation=45)
        plt.savefig('bocp_detector.png')
        change_points4 = detector3.detector(p_value_cutoff=robust_p_value,comparison_window=comparison_window)
        detector3.plot(change_points4)
        plt.xticks(rotation=45)
        plt.savefig('robuststat_detector.png')



        return (change_points1,change_points2,change_points3,change_points4)

    def trends_detection(ts,threshold=0.8,alpha=0.05,direction='up',window_size=20,freq='weekly'):
        df = ts.to_dataframe()
        df_1d = df.set_index('time').resample('1D').sum().iloc[:-1]
        ts_1d = TimeSeriesData(df_1d.reset_index())

        detector = MKDetector(data=ts_1d, threshold=.8)
        # run detector
        detected_time_points = detector.detector(direction=direction,window_size=window_size,freq=freq)
        detector.plot(detected_time_points)
        plt.xticks(rotation=45)
        plt.savefig('trend_detector.png')
        # plot the results
        return detected_time_points

    def build_model(ts,end_day):

        df = ts.to_dataframe().set_index('time').loc[:end_day]
        df = df.tail(20000)
        newts = TimeSeriesData(df.reset_index())
        MD = GetMetaData(data=ts[:], error_method='mape')
        metadata=MD.get_meta_data()

        #best_params = metadata['hpt_res'][metadata['best_model']][0]
        best_params = metadata['hpt_res'][metadata['best_model']]
        best_model = MD.all_models[metadata['best_model']]
        params = MD.all_params[metadata['best_model']](**best_params)

        # import the param and model classes for Prophet model


        return best_model,params


    def predict(model,params,ts,steps=30,freq='30T',include_history=False):

        m = model(ts,params)
        m.fit()
        fcst = m.predict(steps=steps,freq=freq,include_history=include_history)

        m.plot()
        plt.savefig('prediction_results.png')

        
class autogluon:
    
    def read_data(path,brand,freq='30min',prediction_length=48):
        df = pd.read_csv(path,index_col=[0]).reset_index()
        df.columns = ['Date','Value']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').resample(freq).sum()
        df['Brand'] = brand
        
        train_data = TimeSeriesDataFrame.from_data_frame(
        df.reset_index(),
        id_column='Brand',
        timestamp_column='Date'
        )
        
        test_data = train_data.copy() 
        
        train_data = train_data.slice_by_timestep(slice(None, -prediction_length))
        
        return train_data,test_data
    
    
    def build_model(train_data,test_data,target,path,prediction_length,eval_metric='MAPE'):
        predictor = TimeSeriesPredictor(
        path=path,
        target=target,
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        )
        predictor.fit(
            train_data=train_data,
        )
        
        predictor.leaderboard(test_data,silent=True)
        
        return predictor
    
    def predict(predictor,test_data):
        
        predictions = predictor.predict(test_data)
        
        ytrue = test_data.loc['Setur']["Sessions"]
        ypred = predictions.loc['Setur']
        ypred.loc[ytrue.index[-1]] = [ytrue[-1]] * 10
        ypred = ypred.sort_index()
        
        ytrue_test = test_data.loc['Setur']["Sessions"]
        
        plt.plot(ytrue[-30:], label="Training Data")
        plt.plot(ypred["mean"], label="Mean Forecasts")
        plt.plot(ytrue_test, label="Actual")
        
        plt.fill_between(
        ypred.index, ypred["0.1"], ypred["0.9"], color="red", alpha=0.1
        )

        plt.title("Prediction Results")
        _ = plt.legend()
        
        plt.savefig('autogluon_prediction_results.png')
        
    
