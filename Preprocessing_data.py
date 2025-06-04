import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the long format CUSIP data.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)

    # Convert date columns
    date_cols = ['date', 'Maturity Date', 'Issue Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Standardize CUSIP format
    df['cusip_id'] = df['cusip_id'].astype(str)

    # Sort by date and CUSIP
    df = df.sort_values(by=['date', 'cusip_id']).reset_index(drop=True)

    # Calculate time-based features
    df['days_to_maturity'] = (df['Maturity Date'] - df['date']).dt.days

    # Calculate rolling average
    df['rolling_30d_BID'] = (
        df.groupby('cusip_id')['B_YLD_1']
        .transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    )

    # Select relevant columns
    keep_cols = [
        'date', 'cusip_id', 'HIGH_YLD', 'LOW_YLD', 'OPEN_YLD',
        'B_YLD_1', 'BID', 'A_YLD_1', 'ASK_HIGH_1', 'OPEN_ASK', 'ASK',
        'CONVEXITY', 'MOD_DURTN', 'YLDTOMAT', 'YLDWST', 'INT_CDS', 'DIRTY_PRC',
        'Coupon Rate', 'Maturity Date', 'Issue Date', 'rolling_30d_BID', 'days_to_maturity'
    ]
    df = df[keep_cols]

    return df


def calculate_remaining_payments(row):
    """
    Calculate remaining coupon payments until maturity.

    Args:
        row (pd.Series): DataFrame row containing date and maturity information

    Returns:
        int: Number of remaining payments (or np.nan if dates are invalid)
    """
    if pd.isna(row['Maturity Date']) or pd.isna(row['date']):
        return np.nan
    if row['Maturity Date'] < row['date']:
        return 0
    delta_years = (row['Maturity Date'] - row['date']).days / 365.25
    return int(np.floor(delta_years * 2))


def shift_covariates(df):
    """
    Shift covariates one time step back within each CUSIP group.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with shifted covariates
    """
    exclude_cols = ['date', 'cusip_id', 'Issue Date', 'Maturity Date']
    covariate_cols = [col for col in df.columns if col not in exclude_cols]

    df = df.sort_values(by=['cusip_id', 'date']).reset_index(drop=True)
    for col in covariate_cols:
        df[col] = df.groupby('cusip_id')[col].shift(1)

    return df


def load_and_prepare_rfq_data(filepath):
    """
    Load and prepare the aggregated RFQ data.

    Args:
        filepath (str): Path to the RFQ CSV file

    Returns:
        pd.DataFrame: Processed RFQ data
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'trd_exctn_dt': 'date',
        'trd_exctn_tm': 'date_hour'
    })
    df['date'] = pd.to_datetime(df['date'])
    df['cusip_id'] = df['cusip_id'].astype(str)

    keep_cols = [
        'date', 'cusip_id', 'date_hour',
        'total_volume', 'weighted_avg_yield', 'weighted_avg_price'
    ]
    return df[keep_cols]


def perform_cross_sectional_pca(df, pca_cols):
    """
    Perform cross-sectional PCA (one per date) on the specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        pca_cols (list): Columns to use for PCA

    Returns:
        list: List of DataFrames with PCA results
    """
    pca_results = []

    for date, group in df.groupby('date'):
        # Skip dates with insufficient data
        if group[pca_cols].dropna().shape[0] < 5:
            continue

        # Impute and scale data
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X = scaler.fit_transform(imputer.fit_transform(group[pca_cols]))

        # Perform PCA
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X)

        # Create results DataFrame
        pcs_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(4)])
        pcs_df['date'] = group['date'].values
        pcs_df['cusip_id'] = group['cusip_id'].values

        # Preserve other variables
        preserve_cols = [
            'Coupon Rate', 'date_hour', 'remaining_payments',
            'total_volume', 'weighted_avg_yield',
            'weighted_avg_price', 'rolling_30d_BID'
        ]
        for col in preserve_cols:
            if col in group.columns:
                pcs_df[col] = group[col].values

        pca_results.append(pcs_df)

    return pca_results


def plot_sample_cusip(df, cusip_id):
    """
    Plot weighted average yield for a sample CUSIP.

    Args:
        df (pd.DataFrame): Final DataFrame
        cusip_id (str): CUSIP to plot
    """
    if cusip_id in df['cusip_id'].values:
        plt.figure(figsize=(10, 5))
        plt.plot(df[df['cusip_id'] == cusip_id]['weighted_avg_yield'].values)
        plt.title(f"Weighted Average Yield for CUSIP {cusip_id}")
        plt.ylabel("Yield")
        plt.xlabel("Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"CUSIP {cusip_id} not found in the dataset.")


def main():
    # Configuration
    CUSIP_DATA_PATH = "cusip_covariates_long_full.csv"
    RFQ_DATA_PATH = "aggregated_RFQs.csv"
    OUTPUT_PATH = "final_df.csv"
    SAMPLE_CUSIP = "458140BU3"

    # Load and preprocess data
    print("Loading and preprocessing CUSIP data...")
    long_df = load_and_preprocess_data(CUSIP_DATA_PATH)

    # Calculate remaining payments
    print("Calculating remaining payments...")
    long_df['remaining_payments'] = long_df.apply(calculate_remaining_payments, axis=1)

    # Shift covariates
    print("Shifting covariates...")
    long_df = shift_covariates(long_df)

    # Load and merge RFQ data
    print("Loading and merging RFQ data...")
    rfq_df = load_and_prepare_rfq_data(RFQ_DATA_PATH)
    merged_df = pd.merge(long_df, rfq_df, on=['date', 'cusip_id'], how='left')
    merged_df = merged_df.dropna(axis=1, how='all')
    merged_df = merged_df.sort_values(by=['date', 'date_hour']).reset_index(drop=True)

    # Define PCA columns
    pca_cols = [
        'ASK', 'A_YLD_1', 'BID', 'B_YLD_1', 'CONVEXITY',
        'DIRTY_PRC', 'MOD_DURTN', 'YLDTOMAT', 'YLDWST', 'days_to_maturity'
    ]

    # Perform cross-sectional PCA
    print("Performing cross-sectional PCA...")
    pca_results = perform_cross_sectional_pca(merged_df, pca_cols)
    final_df = pd.concat(pca_results, ignore_index=True)

    # Post-processing
    print("Post-processing results...")
    final_df['rolling_30d_BID_scaled'] = StandardScaler().fit_transform(final_df[['rolling_30d_BID']])

    # Normalize percentage columns
    pct_cols = ['Coupon Rate', 'weighted_avg_yield', 'weighted_avg_price']
    for col in pct_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col] / 100

    # Save final results
    final_df = final_df.sort_values(by=['date', 'cusip_id']).reset_index(drop=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")

    # Visualize sample CUSIP
    plot_sample_cusip(final_df, SAMPLE_CUSIP)


if __name__ == '__main__':
    main()