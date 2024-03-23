import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam

# Database connection
engine = create_engine('mssql+pyodbc://YourDatabaseConnectionStringHere')

# SQL Queries to load data
locn_hdr_sql = 'SELECT LOCN_ID, X_COORDINATE, Y_COORDINATE FROM LOCN_HDR'
items_sql = 'SELECT ITEM_ID, PRODUCT_CLASS, WEIGHT, VOLUME FROM ITEMS'
users_sql = 'SELECT USER_ID, DEPARTMENT_CODE, SHIFT_ID, HIRE_DATE FROM USERS'
task_hdr_sql = 'SELECT TASK_ID, TASK_TYPE, USER_ID, EQUIPMENT_ID, CREATED_DATE_TIME, COMPLETED_DATE_TIME FROM TASK_HDR'
task_dtl_sql = 'SELECT TASK_DTL_ID, TASK_ID, SEQ_NBR, PULL_LOCN_ID, ITEM_ID, QTY_PULLD FROM TASK_DTL'

# Load data into DataFrames
locn_hdr_df = pd.read_sql(locn_hdr_sql, engine)
items_df = pd.read_sql(items_sql, engine)
users_df = pd.read_sql(users_sql, engine)
task_hdr_df = pd.read_sql(task_hdr_sql, engine)
task_dtl_df = pd.read_sql(task_dtl_sql, engine)

# Calculate Travel Distance for each task
def calculate_travel_distance(task_details, locations):
    distance = 0
    for i in range(1, len(task_details)):
        loc1 = locations.loc[task_details.iloc[i-1]['PULL_LOCN_ID']]
        loc2 = locations.loc[task_details.iloc[i]['PULL_LOCN_ID']]
        distance += np.sqrt((loc1['X_COORDINATE'] - loc2['X_COORDINATE'])**2 + (loc1['Y_COORDINATE'] - loc2['Y_COORDINATE'])**2)
    return distance

# Merge to get full task details including location coordinates
task_dtl_full_df = task_dtl_df.merge(locn_hdr_df, left_on='PULL_LOCN_ID', right_on='LOCN_ID')

# Group by TASK_ID to calculate Travel Distance
task_travel_distance = task_dtl_full_df.groupby('TASK_ID').apply(lambda x: calculate_travel_distance(x, locn_hdr_df)).reset_index(name='TravelDistance')

# User Experience Level calculation
users_df['HIRE_DATE'] = pd.to_datetime(users_df['HIRE_DATE'])
users_df['ExperienceMonths'] = (pd.to_datetime('today') - users_df['HIRE_DATE']) / np.timedelta64(1, 'M')

# Merge TASK_HDR with USERS to get ExperienceMonths
task_user_experience = task_hdr_df.merge(users_df[['USER_ID', 'ExperienceMonths']], on='USER_ID')

# Calculate other features
task_hdr_df['DayOfWeek'] = task_hdr_df['CREATED_DATE_TIME'].dt.day_name()
task_hdr_df['HourOfDay'] = task_hdr_df['CREATED_DATE_TIME'].dt.hour
task_hdr_df['Month'] = task_hdr_df['COMPLETED_DATE_TIME'].dt.month

# Calculate Total Quantity, Weight, and Volume per Task
task_totals = task_dtl_df.merge(items_df, on='ITEM_ID').groupby('TASK_ID').agg(
    TotalQuantity=pd.NamedAgg(column='QTY_PULLD', aggfunc='sum'),
    TotalWeight=pd.NamedAgg(column='WEIGHT', aggfunc=lambda x: (x * task_dtl_df['QTY_PULLD']).sum()),
    TotalVolume=pd.NamedAgg(column='VOLUME', aggfunc=lambda x: (x * task_dtl_df['QTY_PULLD']).sum())
).reset_index()

# Merge all features with the TASK_HDR dataframe
task_features = task_hdr_df.merge(task_travel_distance, on='TASK_ID') \
                           .merge(task_user_experience, on='TASK_ID') \
                           .merge(task_totals, on='TASK_ID')

# Calculate Completion Time (label)
task_features['CompletionTime'] = (task_features['COMPLETED_DATE_TIME'] - task_features['CREATED_DATE_TIME']).dt.total_seconds() / 60  # in minutes

# Select and rename features for ML model
ml_features = task_features[['TASK_TYPE', 'TravelDistance', 'ExperienceMonths', 'DEPARTMENT_CODE', 'SHIFT_ID', 'DayOfWeek', 'HourOfDay', 'Month', 'EQUIPMENT_ID', 'TotalQuantity', '

# Separate features and target variable
X = ml_features.drop('CompletionTime', axis=1)
y = ml_features['CompletionTime'].values

# Preprocessing
# One-hot encode categorical features and scale numerical features
categorical_features = ['TASK_TYPE', 'DayOfWeek', 'DEPARTMENT_CODE', 'SHIFT_ID', 'EQUIPMENT_ID']
numerical_features = X.columns.difference(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# TensorFlow model
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model on the test set
loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
print(f"Test set Mean Squared Error: {mse:.4f}")
