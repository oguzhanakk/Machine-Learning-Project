import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot


def main():

    sessions= pd.read_excel("selpak.xlsx")
    sessions["Date"] = pd.to_datetime(sessions["Date"], format='%Y-%m-%d %hh:%mm')

    sessions["Date"] = sessions["Date"] + pd.Timedelta(minutes=30)

    sessions.set_index("Date", inplace=True)
    sessions = sessions.resample("60T").sum()
    sessions.fillna(0, inplace=True)

    print(sessions.dtypes)
    result = seasonal_decompose(sessions, model='additive', period=48)
    result.plot()
    pyplot.show()
    sessions["seasonal"] = result.seasonal
    sessions["trend"] = result.trend
    sessions["resid"] = result.resid
    sessions["trend and resid"] = sessions["selpak"] - sessions["seasonal"]
    if min(sessions["trend and resid"]) < 0:
        sessions["incremental selpak trend"] = sessions["trend and resid"] + abs(min(sessions["trend and resid"])) + 1
    else:
        sessions["incremental selpak trend"] = sessions["trend and resid"]

    sessions["New Time"] = sessions.index.time
    sessions["New Date"] = sessions.index.date
    sessions.to_excel("selpak_trend_30.xlsx")


if __name__ == "__main__":
   main()
