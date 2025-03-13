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

# Converts boolean values to '0' (false) or '1' (true) for pre-processing
def convert_booleans(dataframe):
    
    for i in dataframe.columns:
        
        if dataframe[i].dtype == "bool":
            
            dataframe[i] = dataframe[i].apply(lambda x: int(x))
            
        else:
            pass
        
    return dataframe
                 

file_path = "online_shoppers_intention.csv"

# Create dataframe from csv file, dropping rows containing empty values and dropping duplicate rows 
df = pd.read_csv(file_path, delimiter=",", header=0).dropna().drop_duplicates()


cleansed_df = whitespace_remover(df)
processed_df = convert_booleans(cleansed_df)



# Convert dataframe back to csv
processed_df.to_csv('cleaned_data.csv', index=False)



