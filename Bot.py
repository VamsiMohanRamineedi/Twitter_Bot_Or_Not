# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\ramin\OneDrive - nyu.edu\Spring 2017\Machine Learning\Project\train_data.csv', encoding="cp437")
descriptio_list = list(dataset["description"])
#dataset_new=dataset["has_extended_profile"].dropna()
dataset_new = dataset[["screen_name","followers_count","friends_count","listedcount","favourites_count","verified","statuses_count","default_profile","default_profile_image","bot"]]
    