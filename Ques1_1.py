# %%
import pandas as pd  


# %%
df = pd.read_csv("face_image_attr.csv")
df

# %%
# Correlation Heatmap 
import seaborn as sns
import matplotlib.pyplot as plt

df.corr()

# %%
df.head()

# %%
df1 = df[["image_id","Arched_Eyebrows"]]
df1

# %%
testSet = df1[182637:]
testSet

# %%
df1 = df1[:182637]

# %%
testSet["Arched_Eyebrows"].value_counts()

# %%
# Save the test set to a csv file
testSet.to_csv("testSet.csv",index=False)

# %%
df1["Arched_Eyebrows"].value_counts()

# %%
# Extract equal number of rows with Arched_Eyebrows = -1 and Arched_Eyebrows = 1 and choose n rows
df2 = df1[df1["Arched_Eyebrows"] == 1]
df3 = df1[df1["Arched_Eyebrows"] == -1].head(60000)
df1 = pd.concat([df2,df3])


# %%
df1

# %%
df1["Arched_Eyebrows"].value_counts()

# %%
# save this data to a csv file
df1.to_csv("Arched_Eyebrows.csv", index=False)

# %%



