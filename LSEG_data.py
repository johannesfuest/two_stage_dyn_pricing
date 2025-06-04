import pandas as pd


def load_sheets(file_path):
    """
    Load all required sheets from the Excel file.

    Args:
        file_path (str): Path to the Excel file

    Returns:
        tuple: (sheet1, sheet2, sheet4) DataFrames
    """
    return (
        pd.read_excel(file_path, sheet_name='Sheet1', header=None),  # CUSIP list
        pd.read_excel(file_path, sheet_name='Sheet2', header=1),  # Time-series data
        pd.read_excel(file_path, sheet_name='Sheet4', header=1)  # Static data
    )


def extract_cusips(sheet1):
    """
    Extract and clean CUSIP list from Sheet1.

    Args:
        sheet1 (pd.DataFrame): First sheet of the Excel file

    Returns:
        list: Cleaned CUSIP strings
    """
    return sheet1[0].dropna().str.replace('=', '', regex=False).tolist()


def identify_data_blocks(sheet2):
    """
    Identify column ranges for each CUSIP's data block in Sheet2.

    Args:
        sheet2 (pd.DataFrame): Time-series data sheet

    Returns:
        list: List of (start, end) column index tuples for each block
    """
    columns = list(sheet2.columns)
    timestamp_indices = [i for i, col in enumerate(columns) if 'Timestamp' in str(col)]

    return [
        (timestamp_indices[i], timestamp_indices[i + 1]) if i + 1 < len(timestamp_indices)
        else (timestamp_indices[i], len(columns))
        for i in range(len(timestamp_indices))
    ]


def process_time_series_data(sheet2, cusips, block_ranges):
    """
    Process time-series data blocks into a long format DataFrame.

    Args:
        sheet2 (pd.DataFrame): Time-series data sheet
        cusips (list): List of CUSIP identifiers
        block_ranges (list): Column ranges for each CUSIP's data

    Returns:
        pd.DataFrame: Combined time-series data in long format
    """
    if len(cusips) != len(block_ranges):
        raise ValueError("Mismatch between number of CUSIPs and Timestamp blocks.")

    # Get reference columns from first block
    reference_columns = [
        str(col).strip() for col in
        sheet2.iloc[:, block_ranges[0][0]:block_ranges[0][1]].columns
    ]
    reference_columns[0] = 'date'  # Standardize date column name

    records = []
    for cusip, (start, end) in zip(cusips, block_ranges):
        block = sheet2.iloc[:, start:end].copy()
        block.columns = [str(c).strip() for c in block.columns]
        block = block.rename(columns={block.columns[0]: 'date'})
        block.columns = reference_columns
        block.insert(1, 'cusip_id', cusip)
        records.append(block)

    return pd.concat(records, ignore_index=True)


def process_static_data(sheet4, cusips):
    """
    Process static CUSIP data from Sheet4.

    Args:
        sheet4 (pd.DataFrame): Static data sheet
        cusips (list): List of CUSIP identifiers

    Returns:
        pd.DataFrame: Cleaned static data with CUSIP IDs
    """
    sheet4.columns = [str(col).strip() for col in sheet4.columns]
    sheet4 = sheet4.dropna(how='all')  # Remove empty rows
    sheet4['cusip_id'] = cusips[:len(sheet4)]
    return sheet4


def main():
    # Configuration
    INPUT_FILE = "data/LSEG.xlsx"
    OUTPUT_FILE = "cusip_covariates_long_full.csv"

    # Load data
    print("Loading Excel sheets...")
    sheet1, sheet2, sheet4 = load_sheets(INPUT_FILE)

    # Process CUSIPs
    print("Extracting CUSIP list...")
    cusips = extract_cusips(sheet1)

    # Process time-series data
    print("Processing time-series data...")
    block_ranges = identify_data_blocks(sheet2)
    time_series_df = process_time_series_data(sheet2, cusips, block_ranges)

    # Process static data
    print("Processing static data...")
    static_df = process_static_data(sheet4, cusips)

    # Merge datasets
    print("Merging datasets...")
    combined_df = time_series_df.merge(static_df, on='cusip_id', how='left')
    combined_df = combined_df.dropna(axis=1, how='all')  # Remove entirely empty columns

    # Save output
    print(f"Saving results to {OUTPUT_FILE}...")
    combined_df.to_csv(OUTPUT_FILE, index=False)

    # Show preview
    print("\nPreview of final dataset:")
    print(combined_df.head())


if __name__ == '__main__':
    main()