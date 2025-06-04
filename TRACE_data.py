import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the TRACE dataset.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Load the dataset
    trace = pd.read_csv(filepath)

    # Filter out records with empty CUSIP
    trace = trace[trace['cusip_id'] != '']

    # Filter for sell transactions only
    trace = trace[trace['rpt_side_cd'] == 'S']

    return trace


def process_post_feb2012_data(df):
    """
    Process data reported after February 6th, 2012.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Filter for post-Feb 2012 data
    post_feb = df[df['trd_rpt_dt'] >= '2012-02-06'].copy()

    # Split data based on TRC_ST values
    delete_corrections = post_feb[post_feb['trc_st'].isin(['x', 'c'])][
        ['cusip_id', 'entrd_vol_qt', 'rptd_pr', 'trd_exctn_dt',
         'trd_exctn_tm', 'rpt_side_cd', 'cntra_mp_id', 'msg_seq_nb']
    ]

    delete_reversals = post_feb[post_feb['trc_st'].isin(['y'])][
        ['cusip_id', 'entrd_vol_qt', 'rptd_pr', 'trd_exctn_dt',
         'trd_exctn_tm', 'rpt_side_cd', 'cntra_mp_id', 'orig_msg_seq_nb']
    ]

    # Remove corrections and reversals from main dataset
    clean_data = post_feb[~post_feb['trc_st'].isin(['x', 'c', 'y'])]

    # Remove records that were cancelled or corrected
    clean_data = clean_data[~clean_data.index.isin(delete_corrections.index)]

    # Remove records that were reversed
    clean_data = clean_data[~clean_data.index.isin(delete_reversals.index)]

    return clean_data


def process_pre_feb2012_data(df):
    """
    Process data reported before February 6th, 2012.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Filter for pre-Feb 2012 data
    pre_feb = df[df['trd_rpt_dt'] < '2012-02-06'].copy()

    # Remove error trades (corrections and withdrawals)
    error_trades = pre_feb[pre_feb['trc_st'] == 'c'][['trd_rpt_dt', 'orig_msg_seq_nb']]
    clean_data = pre_feb[~pre_feb['trc_st'].isin(['c', 'w'])]
    clean_data = clean_data[~clean_data.index.isin(error_trades.index)]

    # Handle reversals
    reversals = clean_data[clean_data['asof_cd'] == 'r']
    clean_data = clean_data[clean_data['asof_cd'] != 'r']

    # Sort data for proper merging
    reversals = reversals.sort_values(
        by=['trd_exctn_dt', 'cusip_id', 'trd_exctn_tm', 'rptd_pr',
            'entrd_vol_qt', 'rpt_side_cd', 'cntra_mp_id',
            'trd_rpt_dt', 'trd_rpt_tm', 'msg_seq_nb']
    ).drop_duplicates()

    clean_data = clean_data.sort_values(
        by=['trd_exctn_dt', 'cusip_id', 'trd_exctn_tm', 'rptd_pr',
            'entrd_vol_qt', 'rpt_side_cd', 'cntra_mp_id']
    )

    # Merge reversals back and remove matched trades
    matched_reversals = pd.merge(
        clean_data, reversals,
        how='inner',
        on=['trd_exctn_dt', 'cusip_id', 'trd_exctn_tm',
            'rptd_pr', 'entrd_vol_qt', 'rpt_side_cd', 'cntra_mp_id']
    )

    if 'trd_rpt_dt' in matched_reversals.columns:
        matched_reversals = matched_reversals[
            matched_reversals['trd_exctn_dt'] < matched_reversals['trd_rpt_dt']
            ].drop_duplicates(
            subset=['trd_exctn_dt', 'bond_sym_id', 'trd_exctn_tm',
                    'rptd_pr', 'entrd_vol_qt']
        )

    final_data = clean_data[~clean_data.index.isin(matched_reversals.index)]

    return final_data


def filter_agency_transactions(df):
    """
    Filter agency transactions and clean interdealer trades.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()

    # Identify agency transactions
    df['agency'] = df.apply(
        lambda row: row['buy_cpcty_cd'] if row['rpt_side_cd'] == 'b'
        else row['sell_cpcty_cd'],
        axis=1
    )

    # Filter out specific agency transactions if column exists
    if 'cmsn_trd' in df.columns:
        df = df[~(
                (df['agency'] == 'a') &
                (df['cntra_mp_id'] == 'c') &
                (df['cmsn_trd'] == 'n')
        )]

    # Clean interdealer transactions
    df.loc[(df['cntra_mp_id'] == 'd') & (df['rpt_side_cd'] == 'b'), 'rpt_side_cd'] = 'd'
    df = df[~((df['cntra_mp_id'] == 'd') & (df['rpt_side_cd'] == 's'))]

    return df


def aggregate_trades(df):
    """
    Aggregate trades by CUSIP, date, and time.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Aggregated DataFrame with weighted averages
    """
    # Convert to proper datetime formats
    df['trd_exctn_tm'] = pd.to_datetime(df['trd_exctn_tm'], format='%H:%M:%S').dt.time
    df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])

    # Drop rows with missing values in key columns
    df_clean = df.dropna(subset=['entrd_vol_qt', 'yld_pt', 'rptd_pr'])

    # Convert to numeric
    numeric_cols = ['entrd_vol_qt', 'yld_pt', 'rptd_pr']
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric)

    # Group by CUSIP, date, and time to calculate weighted averages
    aggregated = df_clean.groupby(['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm']).apply(
        lambda g: pd.Series({
            'total_volume': g['entrd_vol_qt'].sum(),
            'weighted_avg_yield': (g['yld_pt'] * g['entrd_vol_qt']).sum() / g['entrd_vol_qt'].sum(),
            'weighted_avg_price': (g['rptd_pr'] * g['entrd_vol_qt']).sum() / g['entrd_vol_qt'].sum()
        })
    ).reset_index()

    return aggregated


def main():
    # Configuration
    INPUT_FILE = 'data/trace_enhanced.csv'
    OUTPUT_FILE = 'traceCLEAN.csv'
    AGGREGATED_OUTPUT = 'aggregated_RFQs.csv'

    # Load and preprocess data
    print("Loading and preprocessing data...")
    trace = load_and_preprocess_data(INPUT_FILE)

    # Process data before and after Feb 2012
    print("Processing pre-Feb 2012 data...")
    pre_feb_data = process_pre_feb2012_data(trace)

    print("Processing post-Feb 2012 data...")
    post_feb_data = process_post_feb2012_data(trace)

    # Combine datasets
    print("Combining datasets...")
    combined_data = pd.concat([pre_feb_data, post_feb_data], ignore_index=True)

    # Clean up columns
    columns_to_drop = ['n', 'asof_cd', 'trd_rpt_dt', 'trd_rpt_tm',
                       'msg_seq_nb', 'trc_st', 'orig_msg_seq_nb']
    existing_columns = [col for col in columns_to_drop if col in combined_data.columns]
    combined_data = combined_data.drop(columns=existing_columns)

    # Filter agency transactions
    print("Filtering agency transactions...")
    final_data = filter_agency_transactions(combined_data)

    # Save cleaned data
    print("Saving cleaned data...")
    final_data.to_csv(OUTPUT_FILE, index=False)

    # Print CUSIP counts
    cusip_counts = final_data['cusip_id'].value_counts()
    print("\nCUSIP counts in cleaned data:")
    print(cusip_counts)

    # Aggregate trades
    print("\nAggregating trades...")
    aggregated = aggregate_trades(final_data)
    aggregated.to_csv(AGGREGATED_OUTPUT, index=False)

    # Print aggregated CUSIP counts
    agg_counts = aggregated['cusip_id'].value_counts().reset_index()
    agg_counts.columns = ['cusip_id', 'observation_count']
    print("\nAggregated observation counts by CUSIP:")
    print(agg_counts)


if __name__ == '__main__':
    main()