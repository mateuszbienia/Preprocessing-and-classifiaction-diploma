This project was created for my engineering thesis "SLEEP ANALYSIS BASED ON DATA EXTRACTED FROM A SMART WATCH". Project consists of two modules:
 - preprocessing sleep data,
 - classifiaction of sleep data to recognise sleep state.
---
Preprocessing sleep data from smartwatch uses accelerometer X, Y, Z axis data and heart rate. System aggregates 30 second epoch of measurements into structure that consist of statistical data of the epoch such as variance, kurtosis, min, max and percentiles. Then system removes insignificant features of the datasets. After all operations data is saved on given path.

Additionally tool for calculating missing data from the datasets, for example if dataset is missing few seconds of measurements

---
Classification part consists of creating two classifiers binary and multiclass ones. Binary classifier is used to recognise sleep state from {wake, sleep}. Second classifier recognises sleep stages from {deep, light, REM, wake}. System tests multiple classifiers and multiple parameters using GridSearchCV to find best one with best parameters. Results are presented on plots that shows accuracy of created models.