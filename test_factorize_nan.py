import pandas as pd
import numpy as np
df = pd.DataFrame({'id': ['a', 'b', np.nan, 'a'], 'val': [1, 2, 3, 4], 'w': [0.1, 0.2, 0.3, 0.4]})
print(df.groupby('id')[['val', 'w']].sum())

df_clean = df.dropna(subset=['id', 'val', 'w'])
codes, uniques = pd.factorize(df_clean['id'])
print(codes)
print(uniques)
w_pts_sum = np.bincount(codes, weights=df_clean["val"])
w_sum = np.bincount(codes, weights=df_clean["w"])
sums = pd.DataFrame({
    "id": uniques,
    "val": w_pts_sum,
    "w": w_sum
})
sums.set_index('id', inplace=True)
print(sums)
