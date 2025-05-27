import pandas as pd

# Load all sheets
file_path = "data/LSEG.xlsx"
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
sheet2 = pd.read_excel(file_path, sheet_name='Sheet2', header=1)
sheet4 = pd.read_excel(file_path, sheet_name='Sheet4', header=1)  # Start from 2nd row as header

# === Prepare CUSIP list ===
cusips = sheet1[0].dropna().str.replace('=', '', regex=False).tolist()

# === Extract block columns for each CUSIP from Sheet2 ===
columns = list(sheet2.columns)
timestamp_indices = [i for i, col in enumerate(columns) if 'Timestamp' in str(col)]

block_ranges = [
    (timestamp_indices[i], timestamp_indices[i + 1]) if i + 1 < len(timestamp_indices)
    else (timestamp_indices[i], len(columns))
    for i in range(len(timestamp_indices))
]

if len(cusips) != len(block_ranges):
    raise ValueError("Mismatch between number of CUSIPs and Timestamp blocks.")

# Use first block as reference for column names
reference_columns = [str(col).strip() for col in sheet2.iloc[:, block_ranges[0][0]:block_ranges[0][1]].columns]
reference_columns[0] = 'date'

records = []
for cusip, (start, end) in zip(cusips, block_ranges):
    block = sheet2.iloc[:, start:end].copy()
    block.columns = [str(c).strip() for c in block.columns]
    block = block.rename(columns={block.columns[0]: 'date'})
    block.columns = reference_columns
    block.insert(1, 'cusip_id', cusip)
    records.append(block)

long_df = pd.concat(records, ignore_index=True)

# === Add Sheet4 data (static per CUSIP) ===
sheet4.columns = [str(col).strip() for col in sheet4.columns]
sheet4 = sheet4.dropna(how='all')  # Remove empty rows
sheet4['cusip_id'] = cusips[:len(sheet4)]
long_df = long_df.merge(sheet4, on='cusip_id', how='left')

long_df = long_df.dropna(axis=1, how='all')
long_df.to_csv("cusip_covariates_long_full.csv", index=False)
print(long_df.head())