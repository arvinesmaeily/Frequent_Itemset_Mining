# importing packages
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import time

# generate encoded dataframe plus preprocessing for algorithms
data = pd.read_csv("mushroom.csv")
print(data.head(data.shape[0]))

records = []
for j in range(0, 23):
    for i in range(0, 8124):
        if data.values[i, j] not in records:
            records.append(int(data.values[i, j]))
print(records)

final_df = pd.DataFrame(columns=records)
for i in range(0, 8124):
    for j in range(1, 120):
        final_df.at[i, j] = 0

for i in range(0, 8124):
    for j in range(1, 120):
        if j in data.values[i, :]:
            final_df.at[i, j] = 1
print(final_df.head(final_df.shape[0]))

# apriori algorithm usage
startTime = time.time()
subsets_apriori = apriori(final_df, min_support=0.5, use_colnames=True)
rules_apriori = association_rules(subsets_apriori, metric="lift", min_threshold=1)
subsets_apriori.to_csv("subsets_apriori.csv")
rules_apriori.to_csv("rules_apriori.csv")
print("Time taken {}".format(time.time() - startTime))

# FP-Growth algorithm usage
startTime = time.time()
subsets_fp = fpgrowth(final_df, min_support=0.5, use_colnames=True)
rules_fp = association_rules(subsets_fp, metric="lift", min_threshold=1)
subsets_fp.to_csv("subsets_fpgrowth.csv")
rules_fp.to_csv("rules_fpgrowth.csv")
print("Time taken {}".format(time.time() - startTime))

