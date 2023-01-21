import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_csv("uni_session_1hour.csv")

df['item_id'] = 'H1'
df = df[['item_id', 'timestamp', 'target']]

df['timestamp'] = pd.to_datetime(df['timestamp'])

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
print(train_data)
predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="sMAPE",
)
predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)
predictions = predictor.predict(train_data)
print(predictions.head())
#--------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
# TimeSeriesDataFrame can also be loaded directly from a file

df = pd.read_csv("uni_session_1hour.csv")

df['item_id'] = 'H1'
df = df[['item_id', 'timestamp', 'target']]

df['timestamp'] = pd.to_datetime(df['timestamp'])

#---------------------------------------------------------------------------------
test_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
plt.figure(figsize=(20, 3))
item_id = "H1"
y_past = train_data.loc[item_id]["target"]
y_pred = predictions.loc[item_id]
y_test = test_data.loc[item_id]["target"][-48:]
plt.plot(y_past[-200:], label="Past time series values")
plt.savefig("past_time_series.jpg")
plt.show()
plt.plot(y_pred["mean"], label="Mean forecast")
plt.savefig("mean_forecast.jpg")
plt.show()
plt.plot(y_test, label="Future time series values")
plt.savefig("future_time_series.jpg")
plt.show()
plt.fill_between(
    y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
)
plt.legend()
plt.savefig("10%-90%_confidence_interval.jpg")
plt.show()