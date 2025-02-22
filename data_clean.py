import pandas as pd
import numpy as np


# Removes whitespace from each datapoint                 
def whitespace_remover(dataframe):
    
    for i in dataframe.columns:
        
        if dataframe[i].dtype == "object":
            
            dataframe[i] = dataframe[i].map(str.strip)
            
        else:
            pass 
    
    return dataframe
                 

file_path = "online_shoppers_intention.csv"

# Create dataframe from csv file, dropping rows containing empty values and dropping duplicate rows 
df = pd.read_csv(file_path, delimiter=",", header=0).dropna().drop_duplicates()


cleansed_df = whitespace_remover(df)

# Convert dataframe back to csv
cleansed_df.to_csv('cleaned_data.csv', index=False)



