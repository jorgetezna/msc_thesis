import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

# Define the folder containing the images
image_folder = '/Users/jorgemartinez/thesis_retinanet/datasets/all/rates/TP'

# Initialize dictionaries to count cars per day and per hour
cars_per_day = defaultdict(int)
cars_per_hour = defaultdict(int)

# Regular expression to match the filename pattern
pattern = re.compile(r'.*_(\d{8})_(\d{6})_.*\.jpg')

# Iterate through the files in the folder
for filename in os.listdir(image_folder):
    match = pattern.match(filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        # Extract date and hour
        date = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]
        hour = time_str[:2]
        # Increment the counters
        cars_per_day[date] += 1
        cars_per_hour[hour] += 1

# Divide counts by 2.5 and round to the nearest integer
cars_per_day = {date: round(count / 2.5) for date, count in cars_per_day.items()}
cars_per_hour = {hour: round(count / 2.5) for hour, count in cars_per_hour.items()}

# Convert the dictionaries to pandas DataFrames for better handling
df_cars_per_day = pd.DataFrame(list(cars_per_day.items()), columns=['Date', 'Count'])
df_cars_per_hour = pd.DataFrame(list(cars_per_hour.items()), columns=['Hour', 'Count'])

# Sort the DataFrames
df_cars_per_day['Date'] = pd.to_datetime(df_cars_per_day['Date'])
df_cars_per_day = df_cars_per_day.sort_values('Date')

df_cars_per_hour['Hour'] = df_cars_per_hour['Hour'].astype(int)
df_cars_per_hour = df_cars_per_hour.sort_values('Hour')

# Plot statistics
plt.figure(figsize=(14, 8))

# Plot cars detected per day
plt.subplot(2, 1, 1)
plt.plot(df_cars_per_day['Date'], df_cars_per_day['Count'], marker='o')
plt.title('Number of Cars Detected Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Cars')
plt.xticks(rotation=45)
plt.grid(True)

# Plot cars detected per hour
plt.subplot(2, 1, 2)
plt.bar(df_cars_per_hour['Hour'], df_cars_per_hour['Count'])
plt.title('Number of Cars Detected Per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Cars')
plt.xticks(range(24))
plt.grid(True)

plt.tight_layout()
plt.show()

# Print the DataFrames
print("Cars detected per day:")
print(df_cars_per_day)

print("\nCars detected per hour:")
print(df_cars_per_hour)
