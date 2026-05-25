import pandas as pd
df = pd.read_parquet('US Accidents Model-Free Processed Data.parquet')
df.to_csv(open('US Accidents Model-Free Processed Data.csv','wb+'))