# Predictive Model for Task Completion Time

This repository contains a Python script designed for predicting the completion time of tasks within a warehouse setting, utilizing TensorFlow for building and training a neural network model. The script dynamically fetches and processes data from SQL databases, calculates relevant features for model input, and performs predictive analysis.

## Data Fetching and Preprocessing

The script establishes connections to both cloud and on-premises SQL databases using SQLAlchemy, retrieving data from multiple tables such as `LOCN_HDR`, `ITEMS`, `USERS`, and `TASK_HDR`, along with `TASK_DTL`. These tables contain information about warehouse locations, items, user details, and task headers and details, respectively.

## Feature Engineering

Key features calculated for the model include:
- Travel Distance: The total distance traveled to complete each task.
- User Experience Level: The number of months since the user's hire date.
- Task Characteristics: Day of the week, hour of the day, and month of task creation.
- Task Quantities: Total quantity, weight, and volume of items per task.

## Model Training

A TensorFlow Sequential model is used, comprising an input layer, two dense layers with ReLU activation, and an output layer. The model is compiled with Adam optimizer and mean squared error loss function, trained using the preprocessed and feature-engineered dataset.

## Evaluation

The model's performance is evaluated on a test set, with Mean Squared Error (MSE) as the metric to quantify prediction accuracy.

## Requirements

- pandas
- numpy
- tensorflow
- scikit-learn
- SQLAlchemy

Ensure you have the above Python libraries installed in your environment to execute the script successfully.

## Usage

1. Update the database connection strings `cloud_db_string` and `on_prem_db_string` with your actual cloud and on-premises database connection details.
2. Ensure SQL queries are adjusted according to your database schema.
3. Run the script to fetch data, train the model, and evaluate its performance.

## Contribution

Feel free to fork this repository and contribute by submitting pull requests. Your contributions to improving the predictive accuracy of the model or extending its capabilities are welcome.

## License

Project is open-source and free for personal and commercial use.

This README provides an overview of the script's functionality, data handling, and model training processes, along with instructions for setup and usage.
