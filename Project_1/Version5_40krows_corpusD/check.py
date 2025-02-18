import pandas as pd
file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version5_40krows_corpusD\input_data\filtered_data_2_n1.csv'  # 1-day returns data for testing
filtered_data_2_n1 = pd.read_csv(file_path_d1)
filtered_data_2_n1 = filtered_data_2_n1.dropna(subset=['expected_return', 'actual_return'])

print("\nFinding entries for 2021-12-03 from filtered_data_2_n1:")
filtered_date = filtered_data_2_n1[filtered_data_2_n1['date'] == '2021-12-03']
print("\nFiltered entries:")
print(filtered_date)

# Print detailed information
print("\nDetailed breakdown:")
print(f"Total number of entries: {len(filtered_date)}")
print("\nColumns present:", filtered_date.columns.tolist())
print("\nUnique companies:", filtered_date['Company Name'].unique())
print("\nUnique authors:", filtered_date['author'].unique())

# Print full entry details
print("\nFull entry details:")
for idx, row in filtered_date.iterrows():
    print(f"\nEntry {idx}:")
    for column in filtered_date.columns:
        print(f"{column}: {row[column]}")