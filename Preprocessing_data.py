import vaex
import pandas as pd
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
chunk = False

# This consists in saving the data (because there is millions of rows)
if chunk == True:
    # Divide the data into chunk and then combined the data
    chunk_size = 5_000_000  # Adjust based on memory availability

    with gzip.open('Data/BondDailyDataPublic.csv.gzip', 'rt') as f:
        # Use an iterator to load and process data in chunks
        for i, chunk in enumerate(pd.read_csv(f, chunksize=chunk_size)):
            # Convert chunk to vaex DataFrame
            vaex_df = vaex.from_pandas(chunk)

            # Save to a separate HDF5 file for each chunk or append if you prefer one file
            vaex_df.export_hdf5(f'Data/BondDailyDataPublic_chunk_{i}.hdf5')

    df = vaex.open('Data/BondDailyDataPublic_chunk_*.hdf5')  # The * wildcard loads all files with this pattern
    df.export_hdf5('Data/BondDailyDataPublic_combined.hdf5')


df = vaex.open('Data/BondDailyDataPublic_combined.hdf5')
df = vaex.open('Data/BondDailyDataPublic_chunk_4.hdf5')

df_pandas = df.to_pandas_df()
df_pandas.rename(columns={"cusip_id": "cusip"}, inplace=True)
df_pandas.rename(columns={"trd_exctn_dt": "date"}, inplace=True)
df_pandas['date_m'] = pd.to_datetime(df_pandas['date']).dt.to_period('M').astype(str)
#df_pandas['issuer_cusip'] = df_pandas['cusip'].str[:6]

df_pandas['cusip'].value_counts()