# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ramin\OneDrive - nyu.edu\Spring 2017\Machine Learning\Project\train_data.csv', encoding="cp437")
#descriptio_list = list(dataset["description"])
#dataset_new=dataset["has_extended_profile"].dropna()
dataset_new = dataset[["screen_name","followers_count","friends_count","listedcount","favourites_count","verified","statuses_count","default_profile","default_profile_image","bot"]]
    
X = dataset_new.iloc[:, :-1].values
#y = dataset_new.iloc[:, 9].values

                
# Preprocessing the screen name feature
#dataset_new["screen_name_bool"]=0
import re
pattern=r"bot"
for i in range(0, len(dataset_new)):
    if re.search(pattern, X[i,0]):
        dataset_new["screen_name"][i]=1
        
    else:
        dataset_new["screen_name"][i]=0
    X[i,0]= dataset_new["screen_name"][i]

y = dataset_new.iloc[:, 9].values
