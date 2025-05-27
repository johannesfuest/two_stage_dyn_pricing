import pandas as pd

# Load the dataset
trace = pd.read_csv('data/trace_enhanced.csv')  # Load the uploaded CSV file

# Define input and output dataset names
in_dataset = 'traceIN'
out_dataset = 'traceCLEAN'

# Step 1: Filtering Data Reported After Feb 6th, 2012
temp_raw = trace[trace['trd_rpt_dt'] >= '2012-02-06'].copy()
temp_raw = temp_raw[temp_raw['cusip_id'] != '']

# Filter for sell transactions and Dealer
temp_raw = temp_raw[temp_raw['rpt_side_cd'] == 'S']


# Split data based on conditions
temp_deletei_new = temp_raw[temp_raw['trc_st'].isin(['x', 'c'])][['cusip_id', 'entrd_vol_qt', 'rptd_pr', 'trd_exctn_dt', 'trd_exctn_tm', 'rpt_side_cd', 'cntra_mp_id', 'msg_seq_nb']]
temp_deleteii_new = temp_raw[temp_raw['trc_st'].isin(['y'])][['cusip_id', 'entrd_vol_qt', 'rptd_pr', 'trd_exctn_dt', 'trd_exctn_tm', 'rpt_side_cd', 'cntra_mp_id', 'orig_msg_seq_nb']]
temp_raw = temp_raw[~temp_raw['trc_st'].isin(['x', 'c', 'y'])]

# Deletes the cancellations and corrections as identified by the reports in temp_deletei_new
temp_raw2 = pd.merge(temp_raw, temp_raw[~temp_raw.index.isin(temp_deletei_new.index)], how='inner')

# Deletes the reports that are matched by the reversals
temp_raw3_new = pd.merge(temp_raw2, temp_raw2[~temp_raw2.index.isin(temp_deleteii_new.index)], how='inner')

# Step 2: Filtering Data Reported Before Feb 6th, 2012
temp_raw_pre = trace[trace['trd_rpt_dt'] < '2012-02-06'].copy()
temp_raw_pre = temp_raw_pre[temp_raw_pre['cusip_id'] != '']

# Filter for sell transactions
temp_raw_pre = temp_raw_pre[temp_raw_pre['rpt_side_cd'] == 's']

# Split data based on conditions
temp_delete_pre = temp_raw_pre[temp_raw_pre['trc_st'] == 'c'][['trd_rpt_dt', 'orig_msg_seq_nb']]
temp_raw_pre = temp_raw_pre[~temp_raw_pre['trc_st'].isin(['c', 'w'])]

# Deletes the error trades as identified by the message sequence numbers
temp_raw2_pre = pd.merge(temp_raw_pre, temp_raw_pre[~temp_raw_pre.index.isin(temp_delete_pre.index)], how='inner')

# Take out reversals into a dataset
reversal = temp_raw2_pre[temp_raw2_pre['asof_cd'] == 'r']
temp_raw3_pre = temp_raw2_pre[temp_raw2_pre['asof_cd'] != 'r']

# Sorting the data so that it can be merged
reversal = reversal.sort_values(by=['trd_exctn_dt', 'cusip_id', 'trd_exctn_tm', 'rptd_pr', 'entrd_vol_qt', 'rpt_side_cd', 'cntra_mp_id', 'trd_rpt_dt', 'trd_rpt_tm', 'msg_seq_nb']).drop_duplicates()
temp_raw3_pre = temp_raw3_pre.sort_values(by=['trd_exctn_dt', 'cusip_id', 'trd_exctn_tm', 'rptd_pr', 'entrd_vol_qt', 'rpt_side_cd', 'cntra_mp_id'])

# Merges reversals back on and selects matching observations
reversal2 = pd.merge(temp_raw3_pre, reversal, how='inner', on=['trd_exctn_dt', 'cusip_id', 'trd_exctn_tm', 'rptd_pr', 'entrd_vol_qt', 'rpt_side_cd', 'cntra_mp_id'])

# Ensure that 'trd_rpt_dt' is a column in reversal2
if 'trd_rpt_dt' in reversal2.columns:
    reversal2 = reversal2[reversal2['trd_exctn_dt'] < reversal2['trd_rpt_dt']].drop_duplicates(subset=['trd_exctn_dt', 'bond_sym_id', 'trd_exctn_tm', 'rptd_pr', 'entrd_vol_qt'])
else:
    print("Column 'trd_rpt_dt' not found in reversal2")

# Deletes the matching reversals
temp_raw4_pre = temp_raw3_pre[~temp_raw3_pre.index.isin(reversal2.index)]

# Step 3: Combining the PRE and POST Data
temp_raw_comb = pd.concat([temp_raw4_pre, temp_raw3_new], ignore_index=True)

# Drop columns that exist in the DataFrame
columns_to_drop = ['n', 'asof_cd', 'trd_rpt_dt', 'trd_rpt_tm', 'msg_seq_nb', 'trc_st', 'orig_msg_seq_nb']
existing_columns_to_drop = [col for col in columns_to_drop if col in temp_raw_comb.columns]
temp_raw_comb = temp_raw_comb.drop(columns=existing_columns_to_drop)

# Step 4: Agency Transaction Filtering
temp_raw6 = temp_raw_comb.copy()
temp_raw6['agency'] = temp_raw6.apply(lambda row: row['buy_cpcty_cd'] if row['rpt_side_cd'] == 'b' else row['sell_cpcty_cd'], axis=1)

# Replace 'cmsn_trd' with the actual column name from your dataset
if 'cmsn_trd' in temp_raw6.columns:
    temp_raw6 = temp_raw6[~((temp_raw6['agency'] == 'a') & (temp_raw6['cntra_mp_id'] == 'c') & (temp_raw6['cmsn_trd'] == 'n'))]
else:
    print("Column 'cmsn_trd' not found in temp_raw6")

# Deletes interdealer transactions (one of the sides)
temp_raw6.loc[(temp_raw6['cntra_mp_id'] == 'd') & (temp_raw6['rpt_side_cd'] == 'b'), 'rpt_side_cd'] = 'd'
temp_raw6 = temp_raw6[~((temp_raw6['cntra_mp_id'] == 'd') & (temp_raw6['rpt_side_cd'] == 's'))]

# Compute the number of observations by cusip_id
cusip_counts = temp_raw6['cusip_id'].value_counts()


# Print the counts
print(cusip_counts)


# Step 5: Save the Cleaned Dataset
temp_raw6.to_csv('traceCLEAN.csv', index=False)

test = temp_raw6[temp_raw6['cusip_id'] == "458140BU3"]
test = test[test['cntra_mp_id'] == "D"]


df = temp_raw6
# Ensure time is in the proper format
# Convert time and date to proper formats
df['trd_exctn_tm'] = pd.to_datetime(df['trd_exctn_tm'], format='%H:%M:%S').dt.time
df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])

# Drop rows with missing volume, yield, or price
df_clean = df.dropna(subset=['entrd_vol_qt', 'yld_pt', 'rptd_pr'])

# Convert relevant fields to numeric
df_clean['entrd_vol_qt'] = pd.to_numeric(df_clean['entrd_vol_qt'])
df_clean['yld_pt'] = pd.to_numeric(df_clean['yld_pt'])
df_clean['rptd_pr'] = pd.to_numeric(df_clean['rptd_pr'])

# Group by cusip, date, and time, then aggregate
aggregated = df_clean.groupby(['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm']).apply(
    lambda g: pd.Series({
        'total_volume': g['entrd_vol_qt'].sum(),
        'weighted_avg_yield': (g['yld_pt'] * g['entrd_vol_qt']).sum() / g['entrd_vol_qt'].sum(),
        'weighted_avg_price': (g['rptd_pr'] * g['entrd_vol_qt']).sum() / g['entrd_vol_qt'].sum()
    })
).reset_index()

# Optional: Save to CSV
aggregated.to_csv("aggregated_RFQs.csv", index=False)

# Count number of RFQs (observations) per CUSIP
cusip_counts = aggregated['cusip_id'].value_counts().reset_index()
cusip_counts.columns = ['cusip_id', 'observation_count']

# Display result
print(cusip_counts)