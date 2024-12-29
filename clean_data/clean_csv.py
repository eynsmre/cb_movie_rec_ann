import pandas as pd
import re

xlsx_file = r"C:\Users\yunus\Desktop\ann\clean_data\new_data\5000_5_clean_imdb_data.xlsx"
csv_file = r"C:\Users\yunus\Desktop\ann\clean_data\new_data\latest5000_5_clean_imdb_data.csv"

chunk_size = 10000  # Number of rows per chunk

def clean_whitespace(value):
    """Remove all types of leading/trailing whitespace and normalize spacing."""
    if isinstance(value, str):
        return re.sub(r'\s+', ' ', value.strip())
    return value

with pd.ExcelFile(xlsx_file, engine="openpyxl") as xls:
    sheet_names = xls.sheet_names  # Get all sheet names
    all_data = []  # To store all data in a list

    for sheet_name in sheet_names:
        # Read the entire sheet into a DataFrame
        sheet_df = pd.read_excel(xls, sheet_name=sheet_name)

        # Drop rows with any null values
        sheet_df = sheet_df.dropna()

        # Clean all string columns
        for col in sheet_df.select_dtypes(include=["object"]).columns:
            sheet_df[col] = sheet_df[col].apply(clean_whitespace)

        # Normalize the "Name" column
        if 'Name' in sheet_df.columns:
            sheet_df['Name'] = sheet_df['Name'].str.strip()

        # Remove duplicates based on specific columns
        duplicate_columns = ['Name', 'Description', 'Genre(s)']  # Specify relevant columns
        sheet_df = sheet_df.drop_duplicates(subset=duplicate_columns, keep='first')

        # Remove duplicates based on only the "Name" column
        if 'Name' in sheet_df.columns:
            sheet_df = sheet_df.drop_duplicates(subset=['Name'], keep='first')

        # Process the sheet in chunks and append to all_data list
        for start_row in range(0, sheet_df.shape[0], chunk_size):
            chunk = sheet_df.iloc[start_row:start_row + chunk_size]
            all_data.append(chunk)

# Concatenate all data chunks into a single DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# Write the DataFrame to a CSV file
final_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Conversion completed! CSV file saved as {csv_file}.")
