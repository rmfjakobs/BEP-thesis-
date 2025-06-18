# BEP-thesis-

<pre> ```python
-----------------------------------------------------
#STEP 1 - LOAD & PREPARE BASE DATASET
-----------------------------------------------------
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the data
df_bol = pd.read_excel("DatasetTFF.xlsx")

#Filter for the years 2021-2024, exclude 2025
df_bol = df_bol[df_bol["tff_transportDeadlineDateTime"].dt.year.between(2021,2024)]

#Create a column with only dates (no time) and extract the day of the week (0 = Monday etc.)
df_bol["date"] = df_bol["tff_transportDeadlineDateTime"].dt.date
df_bol["weekday_name"] = df_bol["tff_transportDeadlineDateTime"].dt.day_name()

#Group by date to get the total shipments per day (volume)
df_volume = df_bol.groupby("date").agg({"totalShipments": "sum", "weekday_name": "first"}).reset_index()

--------------------------------------------------------
#STEP 2 - ADD PEAK PERIODS, TEMPERATURE, RETAILER DATA
--------------------------------------------------------

#Define all promotional periods (peak periods from table 6 in the report)
peak_periods = [
    ("2021-03-22", "2021-03-28"),
    ("2021-05-24", "2021-05-30"),
    ("2021-09-27", "2021-10-03"),
    ("2022-03-21", "2022-03-27"),
    ("2022-05-23", "2022-05-29"),
    ("2022-09-26", "2022-09-28"),
    ("2022-10-31", "2022-11-02"),
    ("2023-03-27", "2023-04-02"),
    ("2023-05-22", "2023-05-28"),
    ("2023-09-25", "2023-10-01"),
    ("2024-03-25", "2024-03-31"),
    ("2024-05-20", "2024-05-29"),
    ("2024-09-23", "2024-09-29"),
    ("2024-10-21", "2024-10-27"),
]

#Convert to dates without time
peak_periods = [(datetime.strptime(start, "%Y-%m-%d").date(), datetime.strptime(end,"%Y-%m-%d").date())for start, end in peak_periods]

#The following function will return the peak periods 
def is_peak_period(date):
    return any(start <= date <= end for start, end in peak_periods)

#Apply to volume dataframe
df_volume["is_peak"] = df_volume["date"].apply(is_peak_period).astype(int)

#Temperature data
month_numbers = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

df_temp = pd.read_excel("Temperature.xlsx")
df_temp = df_temp.melt(id_vars = ["Year"], var_name = "month_name", value_name = "temperature") #Reshape the format
df_temp["month"] = df_temp["month_name"].map(month_numbers)
df_temp = df_temp.drop(columns = ["month_name"])
df_temp = df_temp.rename(columns = {"Year": "year"})

#Retailer data
df_retailers = pd.read_excel("Retailers.xlsx") #Add number of active retailers (vvb partners) - same format as the temperature excel file
df_retailers = df_retailers.melt(id_vars = ["Year"], var_name = "month_name", value_name = "vvb_partner_count") #Reshape the format
df_retailers["month"] = df_retailers["month_name"].map(month_numbers)
df_retailers = df_retailers.drop(columns = ["month_name"])
df_retailers = df_retailers.rename(columns = {"Year": "year"})

#Add year and month to the main dataset
df_volume["year"] = pd.to_datetime(df_volume["date"]).dt.year
df_volume["month"] = pd.to_datetime(df_volume["date"]).dt.month
df_volume["month_name"] = pd.to_datetime(df_volume["date"]).dt.month_name()
df_volume = df_volume.merge(df_temp, on = ["year", "month"], how = "left") 
df_volume = df_volume.merge(df_retailers, on = ["year", "month"], how = "left")

#Add warehouse volumes (BFC1 & BFC2)
df_bfc = pd.read_excel("BFC1&2.xlsx")
df_bfc["date"] = pd.to_datetime(df_bfc["tff_deliveryDate"]).dt.date
df_bfc = df_bfc[df_bfc["warehouse"].isin(["BFC1", "BFC2"])]
df_bfc_volume = df_bfc.groupby("date")["orders"].sum().reset_index()
df_bfc_volume = df_bfc_volume.rename(columns = {"orders": "bfc_volume"})
df_volume = df_volume.merge(df_bfc_volume, on = "date", how =  "left")

--------------------------------------------------------
#STEP 3 - CALCULATE THE TFF AND MERGE
--------------------------------------------------------

#Group by date and tff_labels to get the number per label per day
tff_daily = df_bol.groupby(["date", "tff_label"])["totalShipments"].sum().unstack(fill_value = 0)

#Calculate the daily TFF percentages
tff_daily["TFF (%)"] = (tff_daily["ON_TIME"] / tff_daily[["EARLY", "ON_TIME", "LATE"]].sum(axis = 1)) * 100

#Reset index
tff_daily = tff_daily.reset_index()

#Merge into one final dataframe
df_final = df_volume.merge(tff_daily[["date", "TFF (%)"]], on = "date", how = "left")

---------------------------------------------------------------------------------------------------
#STEP 4 - CLEANING, EXTRA FEATURES (BINARY INDICATOR, INTERACTION TERM, ONE-HOT ENCODING), HEATMAP
---------------------------------------------------------------------------------------------------

#Drop rows with missing values for relevant variables
df_model = df_final.dropna(subset = ["totalShipments","temperature", "vvb_partner_count", "TFF (%)", "bfc_volume", "is_peak"])

#Add binary indicator and interaction term
split_date = datetime.strptime("2024-02-26", "%Y-%m-%d").date()
df_model["presplit_bin"] = (df_model["date"] < split_date).astype(int)
df_model["volume_x_presplit"] = df_model["totalShipments"] * df_model["presplit_bin"]

#One-hot encoding weekday and month
df_model = pd.get_dummies(df_model, columns = ["weekday_name", "month_name"], drop_first = True)

#Heatmap to visualize correlations between the variables (no time-dependent or binary indicators) 
plt.figure(figsize = (8,5))
sns.heatmap(df_model[["TFF (%)", "totalShipments", "temperature", "vvb_partner_count", "bfc_volume"]].corr(), annot = True, cmap = "coolwarm_r", vmin = -1, vmax = 1, center = 0)
plt.title("Correlation Matrix", fontweight = "bold")
plt.tight_layout()
plt.show()

-----------------------------------------------------
#STEP 5 - PREPARE DATA FOR NAÏVE AND MOVING AVERAGES 
-----------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Loading the data
df = pd.read_excel("DatasetTFF.xlsx")

#Extract the year from the transport deadline date
df["year"] = df["tff_transportDeadlineDateTime"].dt.year

#Filter for specific years (2021 - 2024)
df = df[df["year"].between(2021,2024)]

#Extract the date
df["date"] = df["tff_transportDeadlineDateTime"].dt.date

#Group by day
daily_tff = df.groupby(["date", "tff_label"])["totalShipments"].sum().unstack(fill_value = 0)

#Calculate the daily TFF percentages
daily_tff["TFF (%)"] = (daily_tff["ON_TIME"] / daily_tff[["EARLY", "ON_TIME", "LATE"]].sum(axis = 1)) * 100

#Reset index and sort by date
daily_tff = daily_tff.sort_values("date").reset_index()

---------------------------------------------
#STEP 6 - DAILY NAïVE FORECAST (1-DAY CYCLE)
---------------------------------------------

#Forward fill to handle non-operational days (e.g. zeros)
daily_tff["TFF_forward_fill"] = daily_tff["TFF (%)"].replace(0, np.nan).ffill()

#Forecast TFF for day t + 1 using actual value from day t
daily_tff["TFF_naive"] = daily_tff["TFF_forward_fill"].shift(1)

#Drop first row and rows where actual TFF is still missing
naive_df = daily_tff.dropna(subset = ["TFF (%)", "TFF_naive"])

#Split the dataset into a training and test set (80/20)
split_index = int(len(naive_df) * 0.8)
train = naive_df.iloc[:split_index]
test = naive_df.iloc[split_index:]

#Calculate performance metrics (test)
mae= mean_absolute_error(test["TFF (%)"], test["TFF_naive"])
rmse = np.sqrt(mean_squared_error(test["TFF (%)"], test["TFF_naive"]))
mape_df = test[test["TFF (%)"] != 0]
mape = np.mean(np.abs((mape_df["TFF (%)"] - mape_df["TFF_naive"]) / mape_df["TFF (%)"])) * 100

#Calculate performance metrics (train)
mae_train = mean_absolute_error(train["TFF (%)"], train["TFF_naive"])
rmse_train = np.sqrt(mean_squared_error(train["TFF (%)"], train["TFF_naive"]))
mape_df = train[train["TFF (%)"] != 0]
mape_train = np.mean(np.abs((mape_df["TFF (%)"] - mape_df["TFF_naive"]) / mape_df["TFF (%)"])) * 100

#Print results
print("MAE_NF_Day:", mae)
print("RMSE_NF_Day:", rmse)
print("MAPE_NF_Day:", mape)

print("MAE_NF_Day_train:", mae_train)
print("RMSE_NF_Day_train:", rmse_train)
print("MAPE_NF_Day_train:", mape_train)

----------------------------------------------------------------------
#STEP 7 - WEEKLY CYCLIC NAïVE FORECAST (SAME WEEKDAY PREVIOUS WEEK) 
----------------------------------------------------------------------

#Extract ISO year, ISO week, and weekday number
daily_tff["iso_year"] = daily_tff["date"].dt.isocalendar().year
daily_tff["iso_week"] = daily_tff["date"].dt.isocalendar().week
daily_tff["weekday"] = daily_tff["date"].dt.weekday  #0 = Monday, 1 = Tuesday etc. 

#shift the current dates forward by 1 day to forecast t + 1
daily_tff["forecast_date"] = daily_tff["date"] + pd.Timedelta(days = 1)
daily_tff["forecast_iso_year"] = daily_tff["forecast_date"].dt.isocalendar().year
daily_tff["forecast_iso_week"] = daily_tff["forecast_date"].dt.isocalendar().week
daily_tff["forecast_weekday"] = daily_tff["forecast_date"].dt.weekday 

#Create dataframe for last week's same weekday
ref_7day = daily_tff[["iso_week", "iso_year", "weekday", "TFF (%)"]].copy()
ref_7day["iso_week"] -= 1 #shift data one week back to take the TFF from the same weekday last week 
ref_7day = ref_7day.rename(columns = {"TFF (%)": "TFF_7day_cyclic"})

cyclic_7day_df = pd.merge(daily_tff, ref_7day, left_on = ["forecast_iso_week", "forecast_iso_year", "forecast_weekday"], right_on = ["iso_week", "iso_year", "weekday"], how = "left").dropna(subset = ["TFF (%)", "TFF_7day_cyclic"])

#Split the dataset into a training and test set (80/20)
split_index = int(len(cyclic_7day_df) * 0.8)
train = cyclic_7day_df.iloc[:split_index]
test = cyclic_7day_df.iloc[split_index:]

#Calculate performance metrics
mae_seasonal = mean_absolute_error(test["TFF (%)"], test["TFF_7day_cyclic"])
rmse_seasonal = np.sqrt(mean_squared_error(test["TFF (%)"], test["TFF_7day_cyclic"]))
mape_seasonal_df = test[test["TFF (%)"] != 0]
mape_seasonal = np.mean(np.abs((mape_seasonal_df["TFF (%)"] - mape_seasonal_df["TFF_7day_cyclic"]) / mape_seasonal_df["TFF (%)"])) * 100

mae_seasonal_train = mean_absolute_error(train["TFF (%)"], train["TFF_7day_cyclic"])
rmse_seasonal_train = np.sqrt(mean_squared_error(train["TFF (%)"], train["TFF_7day_cyclic"]))
mape_seasonal_df = train[train["TFF (%)"] != 0]
mape_seasonal_train = np.mean(np.abs((mape_seasonal_df["TFF (%)"] - mape_seasonal_df["TFF_7day_cyclic"]) / mape_seasonal_df["TFF (%)"])) * 100

#Print results
print("MAE_7day_cyclic:", mae_seasonal)
print("RMSE_7day_cyclic:", rmse_seasonal)
print("MAPE_7day_cyclic:", mape_seasonal)

print("MAE_7day_cyclic_train:", mae_seasonal_train)
print("RMSE_7day_cyclic_train:", rmse_seasonal_train)
print("MAPE_7day_cyclic_train:", mape_seasonal_train)

-------------------------------------------------------------------------
#STEP 8 - WEEEKDAY CYCLIC NAïVE FORECAST (SAME WEEKDAY & WEEK LAST YEAR) 
-------------------------------------------------------------------------
    
#Create a reference dataframe with previous year's TFF
seasonal_ref = daily_tff[["iso_week", "iso_year", "weekday", "TFF (%)"]].copy()

#Shift year forward to match with current year's dates
seasonal_ref["iso_year"] += 1 #take the TFF from same week and same weekday lasta year 
seasonal_ref = seasonal_ref.rename(columns = {"TFF (%)": "TFF_seasonal"})

#Merge to get forecast from same weekday and same week of the previous year
seasonal_df = pd.merge(daily_tff, seasonal_ref, left_on = ["forecast_iso_week", "forecast_iso_year", "forecast_weekday"], right_on = ["iso_week", "iso_year", "weekday"], how = "left").dropna(subset = ["TFF (%)", "TFF_seasonal"])

#Split the dataset into a training and test set (80/20)
split_index = int(len(seasonal_df) * 0.8)
train = seasonal_df.iloc[:split_index]
test = seasonal_df.iloc[split_index:]

#Calculate performance metrics
mae_seasonal = mean_absolute_error(test["TFF (%)"], test["TFF_seasonal"])
rmse_seasonal = np.sqrt(mean_squared_error(test["TFF (%)"], test["TFF_seasonal"]))
mape_seasonal_df = test[test["TFF (%)"] != 0]
mape_seasonal = np.mean(np.abs((mape_seasonal_df["TFF (%)"] - mape_seasonal_df["TFF_seasonal"]) / mape_seasonal_df["TFF (%)"])) * 100

mae_seasonal_train = mean_absolute_error(train["TFF (%)"], train["TFF_seasonal"])
rmse_seasonal_train = np.sqrt(mean_squared_error(train["TFF (%)"], train["TFF_seasonal"]))
mape_seasonal_df = train[train["TFF (%)"] != 0]
mape_seasonal_train = np.mean(np.abs((mape_seasonal_df["TFF (%)"] - mape_seasonal_df["TFF_seasonal"]) / mape_seasonal_df["TFF (%)"])) * 100

#Print results
print("MAE_Seasonal_Naive:", mae_seasonal)
print("RMSE_Seasonal_Naive:", rmse_seasonal)
print("MAPE_Seasonal_Naive:", mape_seasonal)

print("MAE_Seasonal_Naive_train:", mae_seasonal_train)
print("RMSE_Seasonal_Naive_train:", rmse_seasonal_train)
print("MAPE_Seasonal_Naive_train:", mape_seasonal_train)

---------------------------------------------------------------------------
#STEP 9 - YEARLY CYCLIC NAïVE FORECAST (SAME ISO WEEK AVERAGE OF LAST YEAR) 
---------------------------------------------------------------------------
#Create shifted reference: same week, previous year
weekly_seasonal_ref = daily_tff[["iso_week", "iso_year", "TFF (%)"]].copy()

#Shift year forward to match with current year's dates
weekly_seasonal_ref["iso_year"] += 1
weekly_seasonal_ref = weekly_seasonal_ref.rename(columns={"TFF (%)": "TFF_seasonal_week"})

#Merge to get forecast from the same week of the previous year
seasonal_week_df = pd.merge(daily_tff, weekly_seasonal_ref, left_on = ["forecast_iso_week", "forecast_iso_year"], right_on=["iso_week", "iso_year"], how = "left")
seasonal_week_df = seasonal_week_df.dropna(subset=["TFF (%)", "TFF_seasonal_week"])

#Split the dataset into a training and test set (80/20)
split_index = int(len(seasonal_week_df) * 0.8)
train = seasonal_week_df.iloc[:split_index]
test = seasonal_week_df.iloc[split_index:]

#Calculate performance metrics
mae_seasonal_week = mean_absolute_error(test["TFF (%)"], test["TFF_seasonal_week"])
rmse_seasonal_week = np.sqrt(mean_squared_error(test["TFF (%)"], test["TFF_seasonal_week"]))
mape_week_seasonal_df = test[test["TFF (%)"] != 0]
mape_seasonal_week = np.mean(np.abs((mape_week_seasonal_df["TFF (%)"] - mape_week_seasonal_df["TFF_seasonal_week"]) / mape_week_seasonal_df["TFF (%)"])) * 100

mae_seasonal_week_train = mean_absolute_error(train["TFF (%)"], train["TFF_seasonal_week"])
rmse_seasonal_week_train = np.sqrt(mean_squared_error(train["TFF (%)"], train["TFF_seasonal_week"]))
mape_week_seasonal_df = train[train["TFF (%)"] != 0]
mape_seasonal_week_train = np.mean(np.abs((mape_week_seasonal_df["TFF (%)"] - mape_week_seasonal_df["TFF_seasonal_week"]) / mape_week_seasonal_df["TFF (%)"])) * 100

#Print results
print("MAE_NF_Seasonal_Week:", mae_seasonal_week)
print("RMSE_NF_Seasonal_Week:", rmse_seasonal_week)
print("MAPE_NF_Seasonal_Week:", mape_seasonal_week)

print("MAE_NF_Seasonal_Week_train:", mae_seasonal_week_train)
print("RMSE_NF_Seasonal_Week_train:", rmse_seasonal_week_train)
print("MAPE_NF_Seasonal_Week_train:", mape_seasonal_week_train)

--------------------------------------------------------
#STEP 10 - SIMPLE MOVING AVERAGE (7-DAY ROLLING WINDOW) 
--------------------------------------------------------

#Calculate the 7-day SMA and shift it forward to predict day t+1
daily_tff["TFF_MA_7"] = daily_tff["TFF (%)"].rolling(window = 7).mean().shift(1)

#Drop rows where actual or forecast values are missing
ma_df = daily_tff.dropna(subset = ["TFF (%)", "TFF_MA_7"])

#Split the dataset into a training and test set (80/20)
split_index = int(len(ma_df) * 0.8)
train = ma_df.iloc[:split_index]
test = ma_df.iloc[split_index:]

#Calculate performance metrics
mae_ma7 = mean_absolute_error(test["TFF (%)"], test["TFF_MA_7"])
rmse_ma7 = np.sqrt(mean_squared_error(test["TFF (%)"], test["TFF_MA_7"]))
mape_ma7_df = test[test["TFF (%)"] != 0]
mape_ma7 = np.mean(np.abs((mape_ma7_df["TFF (%)"] - mape_ma7_df["TFF_MA_7"]) / mape_ma7_df["TFF (%)"])) * 100

mae_ma7_train = mean_absolute_error(train["TFF (%)"], train["TFF_MA_7"])
rmse_ma7_train = np.sqrt(mean_squared_error(train["TFF (%)"], train["TFF_MA_7"]))
mape_ma7_df = train[train["TFF (%)"] != 0]
mape_ma7_train = np.mean(np.abs((mape_ma7_df["TFF (%)"] - mape_ma7_df["TFF_MA_7"]) / mape_ma7_df["TFF (%)"])) * 100

#Print results
print("MAE_MA7:", mae_ma7)
print("RMSE_MA7:", rmse_ma7)
print("MAPE_MA7:", mape_ma7)

print("MAE_MA7_train:", mae_ma7_train)
print("RMSE_MA7_train:", rmse_ma7_train)
print("MAPE_MA7_train:", mape_ma7_train)

--------------------------------------------------------
#STEP 11 - WEIGHTED MOVING AVERAGE (14-DAY ROLLING WINDOW) 
--------------------------------------------------------

from datetime import timedelta

daily_tff["weekday"] = pd.to_datetime(daily_tff["date"]).dt.day_name()

# Set up a dictionary for easy lookup
tff_dict = daily_tff.set_index("date")["TFF (%)"].to_dict()

# Initialize forecast list
wma_forecast = []
forecast_dates = []

#Prioritize same weekday in lookback
for i in range(14, len(daily_tff)):
    current_date = daily_tff.loc[i, "date"]
    forecast_date = current_date + timedelta(days = 1)
    current_weekday = pd.to_datetime(forecast_date).weekday()  # 0 = Monday etc.

    lookback_days = 14
    values = []
    weights = []

    for j in range(1, lookback_days + 1):
        past_date = current_date - timedelta(days = j)
        if past_date in tff_dict:
            tff_val = tff_dict[past_date]
            values.append(tff_val)
            # Give higher weight if it's the same weekday
            past_weekday = pd.to_datetime(past_date).weekday()
            weight = 13 if past_weekday == current_weekday else 1
            weights.append(weight)

    if values:
        forecast = np.dot(weights, values) / sum(weights)
    else:
        forecast = np.nan

    wma_forecast.append(forecast)
    forecast_dates.append(forecast_date)

#Create dataframe for forecasted values
wma_df = pd.DataFrame({"forecast_date": forecast_dates, "TFF_WMA_forecast": wma_forecast})

#Merge forecast with true values
daily_tff_full = pd.merge(daily_tff, wma_df, left_on = "date", right_on = "forecast_date", how = "left")

#Drop rows with NA values
wma_df_final = daily_tff_full.dropna(subset = ["TFF (%)", "TFF_WMA_forecast"])

#Split the dataset into a training and test set (80/20)
split_index = int(len(wma_df_final) * 0.8)
train = wma_df_final.iloc[:split_index]
test = wma_df_final.iloc[split_index:]

rmse_wma = np.sqrt(mean_squared_error(test["TFF (%)"], test["TFF_WMA_forecast"]))
mae_wma = mean_absolute_error(test["TFF (%)"], test["TFF_WMA_forecast"])
mape_df = test[test["TFF (%)"] != 0]
mape_wma = np.mean(np.abs((mape_df["TFF (%)"] - mape_df["TFF_WMA_forecast"]) / mape_df["TFF (%)"])) * 100

rmse_wma_train = np.sqrt(mean_squared_error(train["TFF (%)"], train["TFF_WMA_forecast"]))
mae_wma_train = mean_absolute_error(train["TFF (%)"], train["TFF_WMA_forecast"])
mape_df = train[train["TFF (%)"] != 0]
mape_wma_train = np.mean(np.abs((mape_df["TFF (%)"] - mape_df["TFF_WMA_forecast"]) / mape_df["TFF (%)"])) * 100

print("MAE_WMA:", mae_wma)
print("RMSE_WMA:", rmse_wma)
print("MAPE_WMA:", mape_wma)

print("MAE_WMA_train:", mae_wma_train)
print("RMSE_WMA_train:", rmse_wma_train)
print("MAPE_WMA_train:", mape_wma_train)

--------------------------------------------
#STEP 12 - Multiple Linear Regression (MLR)
--------------------------------------------
#Select final features
feature_cols = ["totalShipments", "presplit_bin", "volume_x_presplit", "vvb_partner_count", "is_peak", "temperature", "bfc_volume"] + [col for col in df_model.columns if col.startswith("weekday_name") or col.startswith("month_name")]

X = df_model[feature_cols]
y = df_model["TFF (%)"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Scaling of the data to account for large variations 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Fit model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

#Evaluate test data
print("R^2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE:", np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
print(pd.DataFrame({"Variable": feature_cols, "coefficient": model.coef_}))

#Evalaute training data
print("R^2 train:", r2_score(y_train, y_pred_train))
print("MAE train:", mean_absolute_error(y_train, y_pred_train))
print("RMSE train:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
mape_train_df = y_train[y_train != 0]
mape_train_pred = y_pred_train[y_train != 0]
print("MAPE train:", np.mean(np.abs((mape_train_df - mape_train_pred) / mape_train_df)) * 100)

#Statistical significance with statsmodel 
#Add intercept term
X_sm = sm.add_constant(X_train_scaled)

#Fit OLS model
model = sm.OLS(y_train, X_sm).fit()
print(model.summary())

#Residual plot (check for linearity)
residuals = y_test - y_pred
plt.figure(figsize = (8,5))
plt.scatter(y_pred, residuals, alpha = 0.5)
plt.axhline(0, color = "red", linestyle = "--")
plt.xlabel("Predicted TFF (%)", fontsize = 13)
plt.ylabel("Residuals", fontsize = 13)
plt.title("Residual Plot", fontweight = "bold", fontsize = 16)
plt.grid(True)
plt.tight_layout()
plt.show()


#QQ-Plot (check for normality)
residuals = y_test - y_pred
sm.qqplot(residuals, line = "45", fit = True)
plt.title("QQ Plot of Residuals", fontweight = "bold", fontsize = 16)
plt.xlabel("Theoretical Quantiles", fontsize = 13)
plt.ylabel("Sample Quantiles", fontsize = 13)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.grid(True)
plt.tight_layout()
plt.show()


--------------------------------------------
#STEP 13 - Symmetric Uncertainty (SU)
--------------------------------------------

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

#Feature matrix (X) and target (y)
X = df_model[feature_cols].copy()
y = df_model["TFF (%)"].copy()

#Discretize both X and y using KBinsDiscretizer (10 bins used)
discretizer_X = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_disc = discretizer_X.fit_transform(X)

discretizer_y = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
y_disc = discretizer_y.fit_transform(y.values.reshape(-1, 1)).flatten()

#Function to compute symmetric uncertainty (normalized MI)
def symmetric_uncertainty(x, y):
    mi = mutual_info_score(x, y)
    h_x = entropy(np.bincount(x.astype(int)))
    h_y = entropy(np.bincount(y.astype(int)))
    return 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0

#Compute normalized MI for each feature
su_scores = []
for i, col in enumerate(X.columns):
    su = symmetric_uncertainty(X_disc[:, i], y_disc)
    su_scores.append((col, su))

#Create dataframe of results
su_df = pd.DataFrame(su_scores, columns=["Feature", "Symmetric Uncertainty"]).sort_values(by="Symmetric Uncertainty", ascending=False)
print(su_df)

------------------------------------
#STEP 14 - PREPARING DATA FOR ARIMA 
------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_excel("DatasetTFF.xlsx")

# Ensure datetime format
df["tff_transportDeadlineDateTime"] = pd.to_datetime(df["tff_transportDeadlineDateTime"])

#Extract the year from the transport deadline date
df["year"] = df["tff_transportDeadlineDateTime"].dt.year

#Filter for specific years (2021 - 2024)
df = df[df["year"].between(2021,2024)]

# Extract date and group by day to calculate daily TFF
df["date"] = df["tff_transportDeadlineDateTime"].dt.date
daily_tff = df.groupby(["date", "tff_label"])["totalShipments"].sum().unstack(fill_value=0)

# Calculate TFF percentage
daily_tff["TFF (%)"] = (daily_tff["ON_TIME"] / daily_tff[["EARLY", "ON_TIME", "LATE"]].sum(axis=1)) * 100
daily_tff = daily_tff.reset_index()

daily_tff["date"] = pd.to_datetime(daily_tff["date"])
daily_tff = daily_tff.sort_values("date")


------------------------------------
#STEP 15 - TIME SERIES PLOT  
------------------------------------

#7-day rolling average
daily_tff["TFF_rolling"] = daily_tff["TFF (%)"].rolling(window = 7).mean()

# Plotting TFF time series
plt.figure(figsize = (10, 8))
plt.plot(daily_tff["date"], daily_tff["TFF (%)"], label = "TFF (%)", linewidth = 1)
plt.plot(daily_tff["date"], daily_tff["TFF_rolling"], label = "Rolling Average", color = "green")

plt.title("Time Series of Daily TFF (%)", fontsize = 16, fontweight = 'bold')
plt.xlabel("Date", fontsize = 13)
plt.ylabel("TFF (%)", fontsize = 13)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth = 1, bymonthday = 1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation = 45, fontsize = 11)
plt.yticks(fontsize = 11)

plt.axvline(pd.to_datetime("2024-02-26"), color = "red", label = "Volume Split")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Statistically check for trend
import statsmodels.api as sm

y = daily_tff["TFF (%)"].dropna().values
t = np.arange(len(y))

X = sm.add_constant(t)
model = sm.OLS(y,X).fit()
print(model.summary())

---------------------------------------------
#STEP 15 - STATIONARITY TESTS (ADF AND KPSS)  
---------------------------------------------

  
import statsmodels.api as sm

y = daily_tff["TFF (%)"].dropna().values
t = np.arange(len(y))

X = sm.add_constant(t)
model = sm.OLS(y,X).fit()
print(model.summary())


#ADF test (mode 2)

#Drop missing values in TFF
tff_series = daily_tff["TFF (%)"].dropna()
split_index = int(len(tff_series) * 0.8)

#Train-test split (80/20)
train, test = tff_series[:split_index], tff_series[split_index:]

#Run ADF test with intercept (mode 2)
adf_result = adfuller(train, regression = "c")

#Evaluate result
print("ADF Test Statistic (mode 2):", adf_result[0])
print("p-value (mode 2):", adf_result[1])
print("Number of lags used (mode 2):", adf_result[2])
print("Number of observations (mode 2):", adf_result[3])
print("Critical values (mode 2):", adf_result[4])


#KPSS test
kpss_result = kpss(train, regression = "c", nlags = "auto")

print("KPSS Test Statistic (mode 2):", kpss_result[0])
print("p-value (mode 2):", kpss_result[1])
print("Number of lags used (mode 2):", kpss_result[2])
print("Critical values (mode 2):", kpss_result[3])


#Differencing
diff_tff = train.diff().dropna()

adf_diff = adfuller(diff_tff, regression = "c")
print("ADF Test Statistic - diff:", adf_diff[0])
print("p-value - diff:", adf_diff[1])
print("Number of lags used - diff:", adf_diff[2])
print("Number of observations - diff:", adf_diff[3])
print("Critical values - diff:", adf_diff[4])

kpss_diff = kpss(diff_tff, regression = "c", nlags = "auto")
print("KPSS Test Statistic - dff:", kpss_diff[0])
print("p-value - dff:", kpss_diff[1])
print("Number of lags used - dff:", kpss_diff[2])
print("Critical values - dff:", kpss_diff[3])


---------------------------------------------
#STEP 16 - ACF AND PACF PLOTS   
---------------------------------------------

  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize = (8,5))
plot_acf(train, ax = axes[0], lags = 30)
plot_pacf(train, ax = axes[1], lags = 30)
axes[0].set_title("Autocorrelation (ACF) Plot", fontweight = "bold")
axes[1].set_title("Partial Autocorrelation (PACF) Plot", fontweight = "bold")
plt.tight_layout()
plt.show()


---------------------------------------------
#STEP 16 - GRID SEARCH   
---------------------------------------------
  
#Finding the optimal values for ARIMA(p,d,q)

p_values = range(0,8)
q_values = range(0,9)
d = 1

results = []
for p in p_values:
    for q in q_values:
        try:
            model = ARIMA(train, order = (p,d,q)).fit()
            results.append({"p": p, "d": d, "q": q, "AIC": model.aic, "BIC": model.bic})

        except:
            continue

results_df = pd.DataFrame(results).sort_values(by = "BIC").reset_index(drop = True)
print(results_df)



---------------------------------------------
#STEP 16 - FINAL ARIMA MODEL  
---------------------------------------------

#Fit ARIMA(6,1,6) on training data
model = ARIMA(train, order = (6,1,6)).fit()

#Training data
train_pred = model.fittedvalues
train_actual = train.iloc[-len(train_pred):]

#Testing data
forecast = model.get_forecast(steps = len(test))
test_pred = forecast.predicted_mean
test_actual = test

#Evaluate training data
print("MAE training:", mean_absolute_error(train_actual, train_pred))
print("RMSE training:", np.sqrt(mean_squared_error(train_actual, train_pred)))
mape_train_df = train_actual[train_actual != 0]
mape_train_pred = train_pred[train_actual != 0]
print("MAPE training:", np.mean(np.abs((mape_train_df - mape_train_pred) / mape_train_df)) * 100)

print("MAE:", mean_absolute_error(test_actual, test_pred))
print("RMSE:", np.sqrt(mean_squared_error(test_actual, test_pred)))
mape_test_df = test_actual[test_actual != 0]
mape_test_pred = test_pred[test_actual != 0]
print("MAPE:", np.mean(np.abs((mape_test_df - mape_test_pred) / mape_test_df)) * 100)


---------------------------------------------
#STEP 17 - PRE-SPLIT ARIMA MODEL  
---------------------------------------------

daily_tff.set_index("date", inplace = True)

#Train and test on pre-split

split_date = pd.to_datetime("2024-02-26")
tff_presplit = daily_tff[daily_tff.index < split_date]["TFF (%)"].dropna()

#Train-test split (80/20)
split_index = int(len(tff_presplit) * 0.8)
train_pre, test_pre = tff_presplit[:split_index], tff_presplit[split_index:]

#Fit ARIMA(6,1,6) on training data
model = ARIMA(train_pre, order = (6,1,6)).fit()

#Training data
train_pred= model.fittedvalues
train_actual = train_pre.iloc[-len(train_pred):]

#Testing data
forecast = model.get_forecast(steps = len(test_pre))
test_pred = forecast.predicted_mean
test_pred.index = test_pre.index

#Evaluate training data
print("MAE presplit training:", mean_absolute_error(train_actual, train_pred))
print("RMSE presplit training:", np.sqrt(mean_squared_error(train_actual, train_pred)))
mape_train_df = train_actual[train_actual != 0]
mape_train_pred = train_pred[train_actual != 0]
print("MAPE presplit training:", np.mean(np.abs((mape_train_df - mape_train_pred) / mape_train_df)) * 100)

print("MAE presplit:", mean_absolute_error(test_pre, test_pred))
print("RMSE presplit:", np.sqrt(mean_squared_error(test_pre, test_pred)))
mape_test_df = test_pre[test_pre != 0]
mape_test_pred = test_pred[test_pre != 0]
print("MAPE presplit:", np.mean(np.abs((mape_test_df - mape_test_pred) / mape_test_df)) * 100)




---------------------------------------------
#STEP 18 - POST-SPLIT ARIMA MODEL  
---------------------------------------------

#Train and test on pre-split
tff_postsplit = daily_tff[daily_tff.index >= split_date]["TFF (%)"].dropna()

#Train-test split (80/20)
split_index = int(len(tff_postsplit) * 0.8)
train_post, test_post = tff_postsplit[:split_index], tff_postsplit[split_index:]

#Fit ARIMA(6,1,6) on training data
model_post = ARIMA(train_post, order = (6,1,6)).fit()

#Training data
train_pred = model_post.fittedvalues
train_actual = train_post.iloc[-len(train_pred):]

#Testing data
forecast = model_post.get_forecast(steps = len(test_post))
test_pred = forecast.predicted_mean
test_pred.index = test_post.index

#Evaluate training data
print("MAE postsplit training:", mean_absolute_error(train_actual, train_pred))
print("RMSE postsplit training:", np.sqrt(mean_squared_error(train_actual, train_pred)))
mape_train_df = train_actual[train_actual != 0]
mape_train_pred = train_pred[train_actual != 0]
print("MAPE postsplit training:", np.mean(np.abs((mape_train_df - mape_train_pred) / mape_train_df)) * 100)

print("MAE postsplit:", mean_absolute_error(test_post, test_pred))
print("RMSE postsplit:", np.sqrt(mean_squared_error(test_post, test_pred)))
mape_test_df = test_post[test_post != 0]
mape_test_pred = test_pred[test_post != 0]
print("MAPE postsplit:", np.mean(np.abs((mape_test_df - mape_test_pred) / mape_test_df)) * 100)


---------------------------------------------
#STEP 19 - ARIMAX 
---------------------------------------------

#Select final features
feature_cols = ["totalShipments", "presplit_bin", "volume_x_presplit", "vvb_partner_count","is_peak", "temperature", "bfc_volume"] + [col for col in df_model.columns if col.startswith("weekday_name") or col.startswith("month_name")]

# Define target and features
X = df_model[feature_cols].astype(float)
y = df_model["TFF (%)"].astype(float)

# 80/20 chronological split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

---------------------------------------------
#STEP 20 - ARIMAX GRID SEARCH
---------------------------------------------
#Finding the optimal values for ARIMA(p,d,q)

p_values = range(0,8)
q_values = range(0,9)
d = 1

results = []
for p in p_values:
    for q in q_values:
        try:
            model = SARIMAX(y_train, exog = X_train, order = (p,d,q)).fit()
            results.append({"p": p, "d": d, "q": q, "AIC": model.aic, "BIC": model.bic})

        except:
            continue

results_df = pd.DataFrame(results).sort_values(by = "AIC").reset_index(drop = True)
print(results_df)


---------------------------------------------
#STEP 21 - FIT ARIMAX
---------------------------------------------

#Fit ARIMAX model
model_arimax = model = SARIMAX(y_train, exog = X_train, order = (0,1,5)).fit()

#Training data
train_pred = model_arimax.fittedvalues
train_actual = y_train.iloc[-len(train_pred):]

#Testing data
forecast = model_arimax.get_forecast(steps = len(y_test), exog = X_test)
test_pred = forecast.predicted_mean
test_pred.index = y_test.index

#Evaluate training data
print("MAE training:", mean_absolute_error(train_actual, train_pred))
print("RMSE training:", np.sqrt(mean_squared_error(train_actual, train_pred)))
mape_train_df = train_actual[train_actual != 0]
mape_train_pred = train_pred[train_actual != 0]
print("MAPE training:", np.mean(np.abs((mape_train_df - mape_train_pred) / mape_train_df)) * 100)

print("MAE:", mean_absolute_error(y_test, test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))
mape_test_df = y_test[y_test != 0]
mape_test_pred = test_pred[y_test != 0]
print("MAPE:", np.mean(np.abs((mape_test_df - mape_test_pred) / mape_test_df)) * 100)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df_model[["TFF (%)", "totalShipments", "presplit_bin", "volume_x_presplit", "temperature", "vvb_partner_count", "bfc_volume"]].corr(), annot = True, cmap = "coolwarm_r", vmin = -1, vmax = 1, center = 0)
plt.title("Correlation Matrix", fontsize = 16, fontweight = "bold")
plt.show()

---------------------------------------------
#STEP 22 - XGBOOST
---------------------------------------------
#Select final features
feature_cols = ["totalShipments", "presplit_bin", "volume_x_presplit", "vvb_partner_count", "is_peak", "temperature", "bfc_volume"] + [col for col in df_model.columns if col.startswith("weekday_name") or col.startswith("month_name")]

X = df_model[feature_cols]
y = df_model["TFF (%)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

---------------------------------------------
#STEP 22 - XGBOOST GRID SEARCH
---------------------------------------------
#Grid search (hyperparameter tuning)
from sklearn.model_selection import GridSearchCV

#Define the hyperparameter grid
param_grid = {"max_depth" : [4,6,8,10,12], "learning_rate": [0.001,0.01,0.05,0.1], "min_child_weight": [1,3,5,7], "subsample": [0.5,0.8,0.9,1.0], "colsample_bytree": [0.5,0.8,0.9,1.0], "n_estimators": [50,100,150,200,250], "lambda": [1,5,10,20], "alpha": [0,1,5], "gamma": [0,1,5]}
#Create XGBoost model object
xgb_model = xgb.XGBRegressor(objective = "reg:squarederror", random_state = 42)

#Grid Search object
grid_search = GridSearchCV(xgb_model, param_grid, cv = 5, scoring = "neg_mean_squared_error", verbose = 1, n_jobs = -1)

#Fit the GridSearch object to the training data
grid_search.fit(X_train, y_train)

#Print best parameters and score
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

---------------------------------------------
#STEP 22 - XGBOOST MODEL FIT
---------------------------------------------
best_model.fit(X_train, y_train)

y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
mape_test_df = y_test[y_test != 0]
mape_test_pred = y_pred_test[y_test != 0]
print("MAPE:", np.mean(np.abs((mape_test_df - mape_test_pred) / mape_test_df)) * 100)

print("MAE training:", mean_absolute_error(y_train, y_pred_train))
print("RMSE training:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
mape_test_df = y_train[y_train != 0]
mape_test_pred = y_pred_train[y_train != 0]
print("MAPE training:", np.mean(np.abs((mape_test_df - mape_test_pred) / mape_test_df)) * 100)

``` </pre>
