# Feeding Futures
# Dashboard written in Shiny for Python
# Authors: Bryce Comer, Orion Gant, Eden Alem, Carson Coody
# Contributors: ALL HAIL SERVER LORD MICHAEL KOMNICK, COMMANDER OF SECURITY AND OPTIMIZATION


# Imports
from __future__ import annotations
import shiny.experimental as shiny_x
from shiny import App, ui, render, reactive
from htmltools import css
import shinyswatch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from shinywidgets import output_widget, render_widget
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import urllib.request
import csv
import codecs
from datetime import date

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# THIS IS THE DATA POST CLEANING | DO NOT EDIT
# -----------------------------------------------------------------------------------------------------------------------------------------------------

"""# **Reading the Data**"""
# Make use of the google sheet link for accessing the original data
def read_google_sheet(sheetId, sheetName):
  SHEET_ID = sheetId
  SHEET_NAME = sheetName
  url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'

  return url

# replace the google_sheet_token with the token of your desired google sheet from the link. Replace the name_of_sheet with the name of the sheet you want to access.
addresses = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
weather = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
meal_counts = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
unserved_people = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
congvsnoncong = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
event = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
data_2019 = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
region_size = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)
data_2023 = pd.read_csv(read_google_sheet('google_sheet_token', 'name_of_sheet'), index_col=0)

# Make use of the google form link for the sheet for our partner to make use moving forward
new_url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
new_response = pd.read_csv(new_url, index_col = 0)


# change index to be id
addresses = addresses.reset_index()
weather = weather.reset_index()
meal_counts = meal_counts.reset_index()
unserved_people = unserved_people.reset_index()
congvsnoncong = congvsnoncong.reset_index()
event = event.reset_index()
data_2019 = data_2019.reset_index()
region_size = region_size.reset_index()
data_2023 = data_2023.reset_index()

new_response = new_response.reset_index()

"""# **Data Analysis**"""
# print the shape of dataset throughout the years
def years_shape(x, col):
  years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

  shapes = []
  for year in years:
    shapes.append(x[x[col] == year].shape)

  return shapes

# remove timestamp column to drop duplicates
meal_counts = meal_counts.drop('Timestamp', axis=1).drop_duplicates()

# select 7 years of historical data, don't consider 2015
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
meal_counts1 = meal_counts[meal_counts['Year'].isin(years)]

# validate the number of meals to match at the beginning and end of operation at a meal site
def validation1(x):
  before_operation = x['Number of Meals Received / Prepared'] + x['Meals available from previous day']
  after_operation = x['total number of meals served'] + x['Total damaged/incomplete/other non-reimbursable meals'] + x['Total leftover meals']

  return before_operation != after_operation

def validation2(x):
  before_operation = x['Number of Meals Received / Prepared'] + x['Meals available from previous day']
  after_operation = x['Total number of first meals '] + x['Total number second meals'] + x['total program adult meals'] + x['total non-program adult meals'] + x['Total damaged/incomplete/other non-reimbursable meals'] + x['Total leftover meals']

  return before_operation != after_operation

def validation3(x):
  before_operation = x['total number of meals served']
  after_operation = x['Total number of first meals '] + x['Total number second meals'] + x['total program adult meals'] + x['total non-program adult meals']

  return before_operation != after_operation

# flag the errors for records with validation errors
error1 = meal_counts1.apply(validation1, axis=1)
error2 = meal_counts1.apply(validation2, axis=1)
error3 = meal_counts1.apply(validation3, axis=1)
meal_counts1['error'] = error1 | error2 | error3

# select a sub-dataset where errors are not flagged
meal_counts2 = meal_counts1[meal_counts1['error']==False]

# select a sub-dataset where site names and date is not null
meal_counts3 = meal_counts2[(meal_counts2['Site Name'].notnull()) & (meal_counts2['Date'].notnull())]

# drop duplicate records
meal_counts4 = meal_counts3.drop_duplicates(['Site Name', 'Date', 'Meal Type', 'Delivery Time',
        'Number of Meals Received / Prepared',
        'Meals available from previous day', 'Total number of first meals ',
        'Total number second meals', 'total program adult meals',
        'total non-program adult meals', 'total number of meals served',
        'Total damaged/incomplete/other non-reimbursable meals',
        'Total leftover meals'])

# convert date column to be datetime object
meal_counts4['date'] = pd.to_datetime(meal_counts4['Date'], errors='coerce')

# create a week column from the date column
meal_counts4['week_number'] = meal_counts4['date'].dt.isocalendar().week

# merge dataset to include the number of unserved people 
meal_unserved_counts = pd.merge(meal_counts4, unserved_people[['Name Of Site', 'Number of additional children requesting meals after all available meals were served:', 'Date', 'Meal Type']], how='left', left_on=['Site Name', 'Date', 'Meal Type'], right_on=['Name Of Site', 'Date', 'Meal Type'])

# divide the number columns with 5 for non-congregate data
def divide_meals_by_5(counts, meal_numbers):

  for column in meal_numbers:
    counts[column] = counts[column].apply(lambda x: math.ceil(x / 5))

# data cleaning pipeline for congregate data
def data_cleaning_pipeline_cong(df):
  # flag validation error records
  error1 = df.apply(validation1, axis=1)
  error2 = df.apply(validation2, axis=1)
  error3 = df.apply(validation3, axis=1)
  df['error'] = error1 | error2 | error3

  # select a sub-dataset where error is not flagged
  df = df[df['error']==False]

  # select site name and date where its not null
  df1 = df[(df['Site Name'].notnull()) & (df['Date'].notnull())]

  # select needed columns for operation
  df2 = df1[['Site Name', 'Date', 'Meal Type',
       'Number of Meals Received / Prepared',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals',
       'Number of additional children requesting meals after all available meals were served:',
       'error']]

  # drop duplicates
  df3 = df2.drop_duplicates(['Site Name', 'Date', 'Meal Type',
       'Number of Meals Received / Prepared',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals',
       'Number of additional children requesting meals after all available meals were served:'])

  # change date to datetime object and extract week number, year, day of week from date data
  df3['date'] = pd.to_datetime(df3['Date'], errors='coerce')
  df3['week_number'] = df3['date'].dt.isocalendar().week

  df3['Year'] = df3['date'].dt.year
  df3['Day of Week'] = (df3['date'].dt.dayofweek + 1) % 7 + 1

  return df3

# clean the 2019 data using the congregate data cleaning pipeline function
cleaned_2019_data = data_cleaning_pipeline_cong(data_2019)

# data cleaning for non-congregate data
def data_cleaning_pipeline_noncong(df):
  error1 = df.apply(validation1, axis=1)
  error2 = df.apply(validation2, axis=1)
  error3 = df.apply(validation3, axis=1)
  df['error'] = error1 | error2 | error3

  df = df[df['error']==False]

  df1 = df[(df['Site Name'].notnull()) & (df['Date'].notnull())]

  df2 = df1[['Site Name', 'Date', 'Meal Type',
       'Number of Meals Received / Prepared',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals',
       'Number of additional children requesting meals after all available meals were served:',
       'error']]

  df3 = df2.drop_duplicates(['Site Name', 'Date', 'Meal Type',
       'Number of Meals Received / Prepared',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals',
       'Number of additional children requesting meals after all available meals were served:'])

  meal_numbers = ['Number of Meals Received / Prepared',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals',
       'Number of additional children requesting meals after all available meals were served:']

  divide_meals_by_5(df3, meal_numbers)

  df3['date'] = pd.to_datetime(df3['Date'], errors='coerce')
  df3['week_number'] = df3['date'].dt.isocalendar().week

  df3['Year'] = df3['date'].dt.year
  df3['Day of Week'] = (df3['date'].dt.dayofweek + 1) % 7 + 1

  return df3

# clean 2023 data using the data cleaning pipeline function of non-congregate data
cleaned_2023_data = data_cleaning_pipeline_noncong(data_2023)

cleaned_new_response = data_cleaning_pipeline_noncong(new_response)

# select needed columns from earlier merged dataset
meal_unserved_counts = meal_unserved_counts[['Site Name',
       'Date', 'Meal Type',
       'Number of Meals Received / Prepared',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals', 'Year',
       'Day of Week', 'error', 'date', 'week_number',
       'Number of additional children requesting meals after all available meals were served:']]

# append the 2019 and 2023 data
meal_unserved_counts1 = pd.concat([meal_unserved_counts, cleaned_2019_data], ignore_index=True)

meal_unserved_counts2 = pd.concat([meal_unserved_counts1, cleaned_new_response], ignore_index=True)

meal_counts_data = pd.concat([meal_unserved_counts2, cleaned_2023_data], ignore_index=True)

# select meal sites where region information is not null
adddresses_notnull = addresses[addresses['Region'].notna()]
addresses[addresses['Region'].isna()]

# merge region information of each meal site with address information of meal sites
regions = pd.merge(adddresses_notnull[['Name Of Site', 'Address', 'Zip Code', 'County', 'Region']], region_size[['region', 'size']],left_on=['Region'], right_on=['region'])

# merge dataset with merged regions information of meal sites
meals_dataset = pd.merge(meal_counts_data, regions[['Name Of Site', 'Address', 'Zip Code', 'County', 'Region', 'size']], left_on=['Site Name'], right_on=['Name Of Site'])

meal_counts_data[meal_counts_data['Site Name'] == 'Town Creek'].shape

# select needed columns to start conversion to people from our meals datset (we're predicting using people information)
people_counts1 = meals_dataset[['Site Name', 'Date', 'Meal Type',
       'Number of Meals Received / Prepared', 'week_number',
       'Meals available from previous day', 'Total number of first meals ',
       'Total number second meals', 'total program adult meals',
       'total non-program adult meals', 'total number of meals served',
       'Total damaged/incomplete/other non-reimbursable meals',
       'Total leftover meals', 'Year', 'Number of additional children requesting meals after all available meals were served:',
       'Day of Week', 'Address', 'Zip Code', 'County', 'Region', 'size']]

# rename to increase readability
people_counts1.rename(columns={'Site Name':'site_name', 'Date':'date', 'Meal Type' : 'meal_type',
                            'Number of Meals Received / Prepared': 'ordered_meals', 'Meals available from previous day': 'previous_day_meals',
                            'Total number of first meals ': 'first_meals', 'Total number second meals': 'second_meals',
                            'total program adult meals': 'program_adult_meals', 'total non-program adult meals': 'nonprogram_adult_meals',
                            'total number of meals served': 'served_meals', 'Total damaged/incomplete/other non-reimbursable meals': 'damaged_meals',
                            'Total leftover meals': 'leftover_meals', 'Year': 'year', 'Day of Week': 'day_of_week',
                            'Number of additional children requesting meals after all available meals were served:': 'unserved_people',
                            'Address':'address', 'Zip Code': 'zip_code', 'County': 'county',
                            'Region': 'region'}, inplace=True)

# make the default value of unserved people 0
people_counts1['unserved_people'].fillna(0, inplace=True)

# concatenate two or more rows with same number column values for non-congregate data (first step of convering non-cong meal data to people data)
def drop_noncong_duplicates(df):
  numerical_columns = ['ordered_meals', 'previous_day_meals',
          'first_meals', 'second_meals', 'program_adult_meals',
          'nonprogram_adult_meals', 'served_meals', 'damaged_meals',
          'leftover_meals', 'unserved_people']

  aggregations = {col: 'mean' for col in numerical_columns}
  aggregations['meal_type'] = lambda x: 'Breakfast+Lunch'

  return df.groupby(['site_name', 'date', 'year', 'day_of_week', 'week_number', 'address', 'zip_code', 'county', 'region', 'size']).agg(aggregations).reset_index()

# select 2020 data which is non-congregate to begin first conversion step of non-cong data to people data
people_counts1_2020 = people_counts1[people_counts1['year'] == 2020]

people_2020 = drop_noncong_duplicates(people_counts1_2020)

# select the meal sites which were non-congregate in the year 2021
noncong_2021 = ['Beersheba Springs Assembly', 'Coalmont Elementary School', 'Community Action Committee', 'Epiphany Mission Episcopal Church',
'Grundy Housing Authority', 'Monteagle Greene Apartments', 'Morton Memorial United Methodist Church',
'North Elementary School', 'Palmer Elementary School', 'Pelham Elementary School ', 'Sewanee Community Center',
'Swiss Memorial Elementary School ', 'Tracy City Elementary School', 'Christ Church Episcopal']

# perform first step of conversion for part of 2021 data
people_counts1_2021 = people_counts1[(people_counts1['year'] == 2021) & (people_counts1['site_name'].isin(noncong_2021))]

people_2021 = drop_noncong_duplicates(people_counts1_2021)

meal_numbers = ['ordered_meals', 'previous_day_meals',
        'first_meals', 'second_meals', 'program_adult_meals',
        'nonprogram_adult_meals', 'served_meals', 'damaged_meals',
        'leftover_meals', 'unserved_people']

# second step of conversion - dividing non-cong meal data to number of people data by dividing with the bulk amount given out (in this case 5 meal packs)
divide_meals_by_5(people_2020, meal_numbers)
divide_meals_by_5(people_2021, meal_numbers)

# drop 2020 and part of 2021 non-cong data from dataset
people_counts2 = people_counts1.drop(people_counts1[(people_counts1['year'] == 2020)].index)

people_counts3 = people_counts2.drop(people_counts2[(people_counts2['year'] == 2021) & (people_counts2['site_name'].isin(noncong_2021))].index)

# concatenate the converted non-cong data with dataset
people_counts4 = pd.concat([people_counts3, people_2020], ignore_index=True)
people_dataset = pd.concat([people_counts4, people_2021], ignore_index=True)

# PLEASE BE AWARE THAT EVERYTHING MENTIONED HERE IS IN PEOPLE FROM NOW ON NOT MEALS, IGNORE THE VARIABLE NAMES BEING IN MEALS

# add more features
people_dataset['available_meals'] = people_dataset['ordered_meals'] + people_dataset['previous_day_meals']
people_dataset['wasted_meals'] = people_dataset['damaged_meals'] + people_dataset['leftover_meals']

people_dataset1 = people_dataset.sort_values(['year', 'week_number'])
week_id = people_dataset1[['year', 'week_number']].copy()
week_id['week_serial_number'] = week_id.groupby(['year', 'week_number']).ngroup()
week_id[['week_serial_number','week_number','year']]
week_id.drop_duplicates(inplace=True)

# merge id with dataset (preparing for the model)
meals_dataset1 = pd.merge(people_dataset1, week_id[['year', 'week_number', 'week_serial_number']], on=['year', 'week_number'])

# select and create event data; merge with dataset (event data is in binary)
event['date'] = pd.to_datetime(event['date'], errors='coerce')
event['week_number'] = event['date'].dt.isocalendar().week
event['year'] = event['date'].dt.year

meals_dataset2 = pd.merge(meals_dataset1, event[['year', 'week_number', 'event']], how='left', on=['year', 'week_number'])

meals_dataset2['event'].fillna(0, inplace=True)

# select important weather information 
weather['date'] = pd.to_datetime(weather['datetime (UTC)'], errors='coerce')
weather['year'] = weather['date'].dt.year
weather['week_number'] = weather['date'].dt.isocalendar().week

weather_info = weather.groupby(['year', 'week_number'])[['temperature (degF)', 'total_precipitation (mm of water equivalent)', 'wind_speed (m/s)','humidex_index (degF)']].mean().reset_index()

# categorize weather data
threshold_humidity = np.quantile(weather_info['humidex_index (degF)'], [0.25, 0.5, 0.75])
threshold_temp= np.quantile(weather_info['temperature (degF)'], [0.25, 0.5, 0.75])
threshold_prec = np.quantile(weather_info['total_precipitation (mm of water equivalent)'], [0.25, 0.5, 0.75])
threshold_wind = np.quantile(weather_info['wind_speed (m/s)'], [0.25, 0.5, 0.75])

weather_info['humidex_index (degF)1'] = weather_info['humidex_index (degF)'].apply(lambda x: 0 if x <  threshold_humidity[0]
                                                                                  else (1 if (x <  threshold_humidity[1])
                                                                                  else (2 if x <  threshold_humidity[2] else 3)))

weather_info['temperature (degF)1'] = weather_info['temperature (degF)'].apply(lambda x: 0 if x <  threshold_temp[0]
                                                                                  else (1 if (x <  threshold_temp[1])
                                                                                  else (2 if x <  threshold_temp[2] else 3)))

weather_info['total_precipitation (mm of water equivalent)1'] = weather_info['total_precipitation (mm of water equivalent)'].apply(lambda x: 0 if x <  threshold_prec[0]
                                                                                  else (1 if (x <  threshold_prec[1])
                                                                                  else (2 if x <  threshold_prec[2] else 3)))

weather_info['wind_speed (m/s)1'] = weather_info['wind_speed (m/s)'].apply(lambda x: 0 if x <  threshold_wind[0]
                                                                                  else (1 if (x <  threshold_wind[1])
                                                                                  else (2 if x <  threshold_wind[2] else 3)))

# rename to increase readability
weather_info.rename(columns={'temperature (degF)1': 'temperature', 'humidex_index (degF)1': 'humidity',
                             'total_precipitation (mm of water equivalent)1': 'precipitation', 'wind_speed (m/s)1':'wind'}, inplace=True)

weather_info.drop(['temperature (degF)',
       'total_precipitation (mm of water equivalent)', 'wind_speed (m/s)',
       'humidex_index (degF)'], axis=1, inplace=True)

# merge with the final dataset
dataset = pd.merge(meals_dataset2, weather_info, on=['year', 'week_number'])


# ------------------------------------------------------------------------------------------------------------------------------------
# THE PREDICTIVE MODEL SECTION
# ------------------------------------------------------------------------------------------------------------------------------------

"""# **Feature Engineering**"""
# create important additional features
df1 = dataset.groupby(['week_serial_number', 'region'])['day_of_week'].nunique().reset_index(name='number_of_days_operated')
df2 = df1.groupby('week_serial_number')['number_of_days_operated'].sum()

regions_data = dataset.groupby(['week_serial_number']).nunique()['region'].reset_index(name='number_of_regions')

# currently not used in the model
sites_data = dataset.groupby(['week_serial_number']).nunique()['site_name'].reset_index(name='number_of_sites')

temp = pd.merge(dataset, regions_data, on=['week_serial_number'])
df3 = pd.merge(temp, sites_data, on=['week_serial_number'])

# select needed features and remove null values 
df4=df3[['ordered_meals',
       'previous_day_meals', 'first_meals', 'second_meals',
       'program_adult_meals', 'nonprogram_adult_meals', 'served_meals',
       'damaged_meals', 'leftover_meals','unserved_people','region', 'available_meals', 'wasted_meals',
       'week_serial_number', 'temperature', 'wind',
       'humidity', 'precipitation', 'number_of_regions',
       'number_of_sites', 'year', 'event', 'size']]

df4['unserved_people'].fillna(0, inplace=True)

# demand column (to predict)
df4['meals_needed'] = df4['served_meals'] + df4['unserved_people']

df4.fillna(0, inplace=True)

# convert categorical information to numerical binary columns
df6 = pd.get_dummies(data=df4, columns=['region'])

# condense data by week serial number id value level; created during data cleaning 
md1 = df6.groupby('week_serial_number')[['meals_needed', 'event', 'size']].agg({'meals_needed': 'sum', 'event': 'sum', 'size': lambda x: x.unique().sum()}).reset_index()

binary_columns = ['event']
for col in binary_columns:
  md1.loc[md1[col] >= 1, col] = 1

md2 = df6.groupby('week_serial_number')[['number_of_regions', 'number_of_sites', 'year', 'temperature', 'humidity', 'precipitation', 'wind']].mean()

md2['week_order'] = md2.groupby(['year']).cumcount() + 1

md3=df6.drop(['meals_needed','temperature', 'humidity', 'precipitation', 'wind',
                   'number_of_regions', 'number_of_sites', 'year', 'event', 'size'], axis=1)
md4=md3.groupby('week_serial_number').sum().reset_index()

temp=pd.merge(md1,md2, on='week_serial_number')
df7=pd.merge(temp,md4, on='week_serial_number')

df = pd.merge(df7, df2, on=['week_serial_number'])

# create additional features for model prediction (2 weeks prior and last year data to train the model on)
df['meals_needed_2weeks_prior'] = df[['meals_needed']].shift(2)
df['meals_available_2weeks_prior'] = df[['available_meals']].shift(2)
df['meals_served_2weeks_prior'] = df[['served_meals']].shift(2)
df['people_unserved_2weeks_prior'] = df[['unserved_people']].shift(2)

df['meals_needed_2weeks_prior'].fillna(df['meals_needed_2weeks_prior'].mean(), inplace=True)
df['meals_available_2weeks_prior'].fillna(df['meals_available_2weeks_prior'].mean(), inplace=True)
df['meals_served_2weeks_prior'].fillna(df['meals_served_2weeks_prior'].mean(), inplace=True)
df['people_unserved_2weeks_prior'].fillna(df['people_unserved_2weeks_prior'].mean(), inplace=True)

df['prev_year_meals_needed_this_week'] = df.groupby('week_order')['meals_needed'].shift()
df['prev_year_meals_needed_this_week'].fillna(df['prev_year_meals_needed_this_week'].mean(), inplace=True)

# df = df[:59]

#weather forecasts 2 weeks ahead
#number of sites
#ordered meals

"""## **Prepping Data for Prediction**"""
# select features for model prediction
df_subset = df.drop(['number_of_sites', 'year', 'ordered_meals', 'previous_day_meals', 'first_meals', 'second_meals',
       'program_adult_meals', 'nonprogram_adult_meals','damaged_meals', 'leftover_meals','available_meals',
       'served_meals','unserved_people', 'wasted_meals' ], axis=1)

# split dataset into test and training data (training for future prediction)
# test_data = df_subset.tail(10) 
train_data = df_subset

# select needed columns for each train and test data
def model_building(train_data):
  X_train = train_data.drop(['meals_needed', 'week_serial_number'], axis=1)
  y_train = train_data['meals_needed']
  # X_test = test_data.drop(['meals_needed', 'week_serial_number'], axis=1)
  # y_test = test_data['meals_needed']

  return X_train, y_train


X_train, y_train = model_building(train_data)

"""## **Prediction using XGBRegression**"""
# initiate the XGBRegressor class of a regression model
xgbr = XGBRegressor()
# Train the model
xgbr.fit(X_train, y_train)

# predict demand (number of people)
# predictions = xgbr.predict(X_test)

# calculate the error of the predicitons
# mae = mean_absolute_error(y_test, predictions)
# print("Mean Absolute Error: ", mae)

# rmse = mean_squared_error(y_test, predictions, squared=False)
# print("Root Mean Squared Error (RMSE):", rmse)

# mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
# print("Mean Absolute Percentage Error (MAPE):", mape)

"""# **Integration with the Dashboard**"""



# -------------------------------------------------------------------------------------
# This is Orion's code for the graphs. It is for Adding column for weeks in operations.
# -------------------------------------------------------------------------------------

week_number_dict_min = dict(dataset.groupby('year')['week_number'].min())

def week(row):
  return row['week_number'] - week_number_dict_min[row['year']] + 1

# This adds a new column called relative week number which is the relative week number (add more in depth description). MAIN POINT: USED IN ALL GRAPHS
dataset['Relative_Week_Number'] = dataset.apply(week, axis=1)

# This adds a new column called Percentage of wasted, USED IN THE 2nd GRAPH
dataset['Percentage of Wasted'] = (dataset['wasted_meals'] / dataset['available_meals']) * 100

# Making a list for the sites is easier for coding below, makes it cleaner. Might add more lists later if have time (I encourage it).
all_sites = ['All Sites']
all_sites = all_sites + (list(dataset['site_name']))
all_regions = ['All Regions']
all_regions = all_regions + (list(dataset['region']))

all_years = ['All Years']
all_years = all_years + (list(dataset['year']))

max_weeks = dataset['Relative_Week_Number'].max()

# 4th graph stuff
day_name = {2:"Mondays", 3:"Tuesdays", 4:"Wednesdays", 5:"Thursdays", 6:"Fridays", 7:"Saturdays", 1:"Sundays"}

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# THIS IS THE UI OF THE DASHBOARD. IF YOU DO NOT KNOW WHAT YOU ARE DOING, DO NOT TOUCH!
# -----------------------------------------------------------------------------------------------------------------------------------------------------

app_ui = ui.page_fluid(
    shinyswatch.theme.pulse(),
    # This creates a fluid page with a title of "DataLab: Feeding Futures".
    ui.panel_title("", "DataLab: Feeding Futures"),
    # This creates a card header with the text "DataLab"
    shiny_x.ui.card_header("DataLab", style="font-size: 36px; color: white; background-color: rgba(95, 54, 110); text-align: right"),
    # This creates a navset tab with a nav item for the homepage.
    ui.navset_tab(
        ui.nav("Homepage", shiny_x.ui.card(
             # This creates a card with a header, title, body, image, and footer.
            shiny_x.ui.card_header("Feeding Futures", class_="text-center", style="font-size: 24px; color: white; background-color: rgba(95, 54, 110)"),
            shiny_x.ui.card_title("Documentation"),
            ui.p("Insights Tab: This tab lets you explore the data by year, region, and site. You can choose one or more options from each of these categories. To do that, follow these steps: Click on the box of the category you want to change (for example, Year). Press backspace to delete the current input. You will see a list of available options. You can select one or more of them. Repeat for the other categories if needed. The categories are reactive, which means they depend on each other. For example, if you select a year, you will only see the regions and sites that were active in that year. If you select a region, you will only see the sites that were active in that region. The tab shows four graphs based on your selections. The first three graphs show the data for weeks one to eight. The last graph shows the average data for each weekday. If a graph is incomplete, it means there was not enough data for that combination of year, region, and site. If you see any errors (red text), it means there was a problem with the data source, not with the code. We cannot fix these errors on our end."),
            # shiny_x.ui.card_image(
            #     file=None,
            #     src="",
            # ),
            ui.p("Oracle Tab: This tab lets you predict the number of people who will show up based on four variables. You can choose one or more options for each variable. To do that, follow these steps: Click on the box of Select Regions and choose the regions you want to predict for. You can select as many as you want. Click on the box of Local Event / Major Holiday and choose if there is any event or holiday that might affect the attendance. The model will use historical data and your other inputs to estimate the impact. You don’t need to specify what kind of event or holiday it is. Click on the box of Total number of days you will be working in all regions combined during that week and choose how many days you will work. Click on the box of Current Week and choose which week of the summer it is. The model can handle up to week 53 in case the program is extended. After you make your selections, the model will run and show you a number as the predicted attendance."),
            shiny_x.ui.card_footer("", class_="text-center", style="background-color: rgba(95, 54, 110); color: white; font-size: 18px"),
            full_screen=True,
        )),
        ui.nav("Insights", 
            # This creates a nav item for the Insights section.
            ui.row(
                ui.column(3,
                    # The user can select a year, week, and site to filter the data.
                    ui.input_selectize(id="year", label="Year:", choices=all_years, multiple=True, selected='All Years'),
                    ui.input_selectize(id="region", label="Region:", multiple=True, selected='All Regions', choices=(all_regions)),
                    ui.input_selectize(id="site", label="Site:", choices=all_sites, multiple = True, selected = "All Sites"),
                    # ui.input_selectize(id="week", label="Week:", choices=["All Weeks",1,2,3,4,5,6,7,8],multiple=True, selected='All Weeks'),     
                ),
                # This displays the graph for served and unserved
                ui.column(9,
                    ui.output_plot("served_unserved"),
                ),
            ),

            # This displays the graph for percentage_wasted_meals_per_week
            ui.row(
                ui.column(9,
                    ui.output_plot("percentage_wasted_meals_per_week"),
                    offset = 3
                ),
            ),
            
            # This displays the graph for week_number_vs_meals_ordered
            ui.row(
                ui.column(9,
                    ui.output_plot("week_number_vs_meals_ordered"),
                    offset = 3
                ),
            ),
            
            # This displays the graph for week_day_meal_number
            ui.row(
                ui.column(9,
                    ui.output_plot("week_day_meal_number"),
                    offset = 3
                ),
            ),
        ),
              # This creates a nav item for the Region section.
                # The user can select all regions or a specific region to filter the data.
        ui.nav("Oracle", shiny_x.ui.card(
            ui.input_checkbox_group(id='list_of_regions',label="Select Regions:", choices=(all_regions)),

            ui.input_checkbox("event_true", "Local Event / Major Holiday that week (select if true): ", False),

            # ui.input_numeric("days_operated_in_region", "Total number of days you will be working in all regions combined during that week:", value=0, min=0),
            ui.input_numeric("days_operated_in_region", "Total number of operations in all regions combined during that week:", value=0, min=0),

            ui.input_numeric("current_week", "The week you want to predict for:", value=0, min=1, max=53),
            

            ui.input_action_button("submit", "Click for Predictions"),

            ui.output_text_verbatim("predict_userinput"),

            ui.output_plot('feature_plot'),

            full_screen=False,
        )),
        ui.nav_spacer()
))




# -----------------------------------------------------------------------------------------------------------------------------------------------------
# THIS IS THE SERVER FOR THE DASHBOARD, IF YOU TOUCH THIS NOTHING WILL WORK. DO NOT TOUCH.
# -----------------------------------------------------------------------------------------------------------------------------------------------------

def server(input, output, session):

    site_meals_served_year = reactive.Value(dataset)
    site_meals_served_region = reactive.Value(dataset)
    site_meals_served_site = reactive.Value(dataset)

    @reactive.Effect
    @reactive.event(input.year)
    def regions_in_year():
        if not input.year():
          pass
        elif "All Years" in input.year():
            site_meals_served_year.set(dataset)
            ui.update_selectize(
                "region",
                choices = all_regions,
                selected = "All Regions"
            )
        else:
            site_meals_served_year.set(dataset[dataset['year'].isin(int(n) for n in input.year())])
            new_choices = ['All Regions']
            new_choices = new_choices + list(site_meals_served_year.get()['region'].unique())
            ui.update_selectize(
                "region",
                choices = new_choices,
                selected = "All Regions"
            )

    @reactive.Effect
    @reactive.event(input.year, input.region)
    def sites_in_region():
        if not input.region():
          pass
        elif "All Regions" in input.region():
            site_meals_served_region.set(site_meals_served_year.get())
            ui.update_selectize(
                "site",
                choices = all_sites,
                selected = "All Sites"
            )
        else:
            site_meals_served_region.set(site_meals_served_year.get()[site_meals_served_year.get()['region'].isin(input.region())])
            new_choices = ['All Sites']
            new_choices = new_choices + list(site_meals_served_region.get()['site_name'].unique())
            ui.update_selectize(
                "site",
                choices = new_choices,
                selected = "All Sites"
            )
    
    @reactive.Effect
    @reactive.event(input.year, input.region, input.site)
    def sites():
        if not input.site():
          pass
        elif "All Sites" in input.site():
            site_meals_served_site.set(site_meals_served_region.get())
        else:
            site_meals_served_site.set(site_meals_served_region.get()[site_meals_served_region.get()['site_name'].isin(input.site())])

# EVERYTHING DOWN HERE IS THE RENDER OF THE PLOTS

    @output
    @render.plot
    def served_unserved():

        plot_data = site_meals_served_site.get()
        plot_data = plot_data.groupby('Relative_Week_Number')[['served_meals', 'unserved_people']].mean().reset_index()
        
        # Debuggin tools
        # print("\nUnique Week Numbers:\n")
        # print(plot_data['Relative_Week_Number'].unique())
        # print("\n\nUnique Served Meals Numbers:\n")
        # print(plot_data['served_meals'].unique())
        # print("\n\nUnique Unserved People Numbers:\n")
        # print(plot_data['unserved_people'].unique())

        # This is the plot for the served vs unserved people
        fig = plt.figure(figsize=(8,8))
        sns.lineplot(data=plot_data, x='Relative_Week_Number', y='served_meals', label='People Served',figure=fig)
        sns.lineplot(data=plot_data, x='Relative_Week_Number', y='unserved_people', label='Unserved People',figure=fig)
        plt.xlabel('Week of Operation')
        plt.ylabel('Number of People')
        plt.title('Number of people Unserved and Served vs Week Number')
        plt.xlim(0, max_weeks)
        return fig

    @output
    @render.plot
    def percentage_wasted_meals_per_week():

      plot_data = site_meals_served_site.get()
      plot_data = plot_data.groupby('Relative_Week_Number')['Percentage of Wasted'].mean().reset_index()

      fig = plt.figure(figsize=(8,8))
      sns.barplot(data=plot_data, x='Relative_Week_Number', y='Percentage of Wasted', label='Wasted', figure=fig)
      plt.xlabel('Week of Operation')
      plt.ylabel('Percentage of Wasted Meals')
      plt.title('Percentage of Wasted Meals per Week Number')
      plt.xlim(-1, max_weeks)
      return fig

    @output
    @render.plot
    def week_number_vs_meals_ordered():

      plot_data = site_meals_served_site.get()
      plot_data = plot_data.groupby('Relative_Week_Number')['ordered_meals'].mean().reset_index()

      fig = plt.figure(figsize=(8,8))
      sns.lineplot(data=plot_data, x='Relative_Week_Number', y='ordered_meals', figure=fig)
      plt.xlabel('Week of Operation')
      plt.ylabel('Number of Meals Ordered')
      plt.title('Week Number vs Number of Meals Ordered')
      plt.xlim(0, max_weeks)
      return fig

    @output 
    @render.plot 
    def week_day_meal_number():
      site_meals_melted = pd.melt(site_meals_served_site.get(), id_vars = ['date', 'day_of_week', 'year', 'region'], value_vars = ['wasted_meals', 'served_meals', 'unserved_people'])
      
      site_meals_melted['text_days']=site_meals_melted['day_of_week'].apply(lambda x: day_name[x])
      site_meals_melted=site_meals_melted.sort_values('day_of_week', ascending=True)
      
      fig = plt.figure(figsize=(12,8), dpi=80)
      sns.barplot(data = site_meals_melted, x = site_meals_melted['text_days'], y = site_meals_melted['value'], hue = site_meals_melted['variable'], errorbar = None, figure = fig)
      plt.legend(labels = ['Wasted Meals', 'Unserved Meals', 'Served Meals'])
      plt.xlabel('Weekday')
      plt.ylabel('Number of Meals')
      plt.title('Weekday vs Number of Meals')
      return fig

    # weather api function to get the weather information in 2 weeks
    def weather_func():
      try:
        ResultBytes = urllib.request.urlopen("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Sewanee?include=fcst%2Cobs%2Chistfcst%2Cstats%2Cdays&key=3QCZSDMUE4G8VWNUXHY7ZZEMN&options=beta&contentType=csv")

        # Parse the results as CSV
        CSVText = csv.reader(codecs.iterdecode(ResultBytes, 'utf-8'))

        # Skip the header row
        next(CSVText)

        # Process each row of the CSV data
        d=defaultdict(list)
        for row in (CSVText):
            d['date'].append(row[1])
            d['temperature'].append(row[4])
            d['humidity'].append(row[9])
            d['precipitation'].append(row[10])
            d['wind'].append(row[17])

      except urllib.error.HTTPError as e:
          ErrorInfo = e.read().decode()
          print('Error code:', e.code, ErrorInfo)
          sys.exit()
      except urllib.error.URLError as e:
          ErrorInfo = e.read().decode()
          print('Error code:', e.code, ErrorInfo)
          sys.exit()

      del(d['date'])
      return d
    
    def padding_number(test_data):
      values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700]

      predicted_meals = list(test_data['predicted_meals'])
      meals_needed = list(test_data['meals_needed'])

      n = len(predicted_meals)
      padding = [0] * n
      for i in range(n):
        for j in range(n):
          if (meals_needed[i] >= (predicted_meals[i] - values[j])) and (meals_needed[i] <= (predicted_meals[i] + values[j])):
            padding[i] += 1

      print(padding)


    @output
    @render.text
    @reactive.event(input.submit)
    def predict_userinput():
      # Input from user
      regions_list = input.list_of_regions()
      event_toggle = 1 if input.event_true() else 0
      days_operated = input.days_operated_in_region()
      order_week = input.current_week()

      # Weather forecast results 2 weeks in advance
      d = weather_func()

      for i in d.keys():
        d[i]=[float(j) for j in d[i]]

      # Dictionary for constructing user input (feasible for the model)
      unseen_data={}


      unseen_data['temperature']=np.mean(np.array(d['temperature'][7:]))

      unseen_data['humidity']=np.mean(np.array(d['humidity'][7:]))

      unseen_data['wind']=np.mean(np.array(d['wind'][7:]))

      unseen_data['precipitation']=np.mean(np.array(d['precipitation'][7:]))

      def region_cols(input_list):
        d={}
        regions=['region_Altamont',
             'region_Beersheba Springs', 'region_Coalmont',
             'region_Coalmont-Altamont', 'region_Decherd',
             'region_Downtown Winchester', 'region_Gruetli-Palmer', 'region_Midway',
             'region_Monteagle', 'region_Pelham', 'region_Rural Decherd',
             'region_Sewanee', 'region_Sherwood', 'region_Tracy City',
             'region_Winchester']
        for i in regions:
          if i in input_list:
            d[i]=1
          else:
            d[i]=0
        return d

      # Required input from user
      unseen_data['event']= event_toggle
      unseen_data['week_order']= order_week
      unseen_data['number_of_regions']= len(regions_list)
      unseen_data['number_of_days_operated']= days_operated

      last_yr = date.today().year - 1

      unseen_data['prev_year_meals_needed_this_week']=int(df[(df['year']==last_yr) & (df['week_order']==unseen_data['week_order'])]['meals_needed'])
      unseen_data['meals_needed_2weeks_prior']=float(df[df['week_serial_number']==df['week_serial_number'].max()]['meals_needed'])
      unseen_data['meals_available_2weeks_prior']=float(df[df['week_serial_number']==df['week_serial_number'].max()]['available_meals'])
      unseen_data['meals_served_2weeks_prior']=float(df[df['week_serial_number']==df['week_serial_number'].max()]['served_meals'])
      unseen_data['people_unserved_2weeks_prior']=float(df[df['week_serial_number']==df['week_serial_number'].max()]['unserved_people'])

      input_list1=['region_'+i for i in regions_list]
      region_dict=region_cols(input_list1)

      unseen_data['size'] = 0

      for i in regions_list:
        size = region_size[region_size['region'] == i]['size']
        unseen_data['size'] += size


      test_d = {**unseen_data, **region_dict}
      x_test=pd.DataFrame(test_d, index=[0])

      # categorize the weather data
      heu_temperature = x_test['temperature']
      heu_humidity = x_test['humidity']
      heu_wind = x_test['wind']
      heu_precipitation = x_test['precipitation']


      x_test['temperature']=x_test['temperature'].apply(lambda x: 0 if x <  threshold_temp[0]
                                                                                        else (1 if (x <  threshold_temp[1])
                                                                                        else (2 if x <  threshold_temp[2] else 3)))

      x_test['humidity']=x_test['humidity'].apply(lambda x: 0 if x <  threshold_humidity[0]
                                                                                        else (1 if (x <  threshold_humidity[1])
                                                                                        else (2 if x <  threshold_humidity[2] else 3)))

      x_test['wind']=x_test['wind'].apply(lambda x: 0 if x <  threshold_wind[0]
                                                                                        else (1 if (x <  threshold_wind[1])
                                                                                        else (2 if x <  threshold_wind[2] else 3)))
      x_test['precipitation']=x_test['precipitation'].apply(lambda x: 0 if x <  threshold_prec[0]
                                                                                        else (1 if (x <  threshold_prec[1])
                                                                                        else (2 if x <  threshold_prec[2] else 3)))



      x_test=x_test[X_train.columns]

      # predict the number of people
      prediction_result = xgbr.predict(x_test)

      # Display these top five important features with a label
      feat_importances = pd.DataFrame(xgbr.feature_importances_, index=X_train.columns).reset_index().sort_values(0,ascending=False)
      first_five = feat_importances.head(5)
      print(first_five)
      print(type(first_five))

      # Predicted weather to display

      print(prediction_result)

      total_meals = dataset['served_meals'].sum()
      region_meals = dataset.groupby('region')['served_meals'].sum()
      percent_distribution = (region_meals / total_meals)
      percent_distribution.reset_index()

      # classify to different regions by percent
      distribution = []
      for value in zip(percent_distribution.index, percent_distribution):
        distribution.append((value[0], value[1] * prediction_result, value[1]))

      print(distribution)
      total_percent = 0
      distributions = []
      for name, pr, percent in distribution:
        if name in regions_list:
          total_percent += round(float(percent), 2)

      for name, pr, percent in distribution:
        if name in regions_list:
          distributions.append((name, round((round(float(percent), 2))/total_percent, 2)))

      print(distributions)

      output_string = ''
      for name, pr in distributions:
        output_string += f"\nRegion: {name}: {pr * 100}%"

      # Function we used to determine padding number in this case 300 is the function defined earlier named padding_number()
      output = f'''
      Predicted People: {max(0, math.ceil(prediction_result)-300)} - {math.ceil(prediction_result)+300}\n
      Temperature: {float(round(heu_temperature, 2))}\n
      Humidity: {float(round(heu_humidity, 2))}\n
      Wind: {float(round(heu_wind, 2))}\n
      Precipitation: {float(round(heu_precipitation, 2))}\n
      Percentage Distribution Accross Regions: {output_string}\n
      Top 5 Features in Order of Importance:\n
      '''
      
      
      return output

    # display graph of feature importance labeled by the model
    @output
    @render.plot
    def feature_plot():
      feat_importances = pd.DataFrame(xgbr.feature_importances_, index=X_train.columns).reset_index().sort_values(0,ascending=False)
      first_five = feat_importances.head(5)

      fig = plt.figure(figsize=(8,8))
      fig = plt.barh(first_five['index'], first_five[0])
      return fig


app = App(app_ui, server)
