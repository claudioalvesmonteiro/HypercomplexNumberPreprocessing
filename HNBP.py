import math
import numpy as np
import pandas as pd


class HypercomplexNumberPreprocessing():

    def __init__(self, mode):

        self.mode = mode


    def angle(self, vector):

        if len(vector) % 2 == 1:
            vector = np.append(vector, 1)
        
        output_vector = [math.atan(vector[i]/vector[i+1]) for i in range(0, len(vector), 2)] +     [np.sqrt(sum([i*i for i in vector])), math.asin(vector[-1]/np.sqrt(sum([i*i for i in vector])))] 
        
        return output_vector



    def radius(self, vector):
    
        if len(vector) % 2 == 1:
            vector = np.append(vector, 1)
        
        output_vector = [math.sqrt(vector[i]**2 + vector[i+1]**2) for i in range(0, len(vector), 2)] + [np.sqrt(sum([i*i for i in vector])), math.asin(vector[-1]/np.sqrt(sum([i*i for i in vector])))] 
        
        return output_vector


    def angleradius(self, vector):
        
        if len(vector) % 2 == 1:
            vector = np.append(vector, 1)
        
        output_vector = [math.sqrt(vector[i]**2 + vector[i+1]**2) for i in range(0, len(vector), 2)] + [math.atan(vector[i]/vector[i+1]) for i in range(0, len(vector), 2)] + [np.sqrt(sum([i*i for i in vector])), math.asin(vector[-1]/np.sqrt(sum([i*i for i in vector])))]               
        
        return output_vector



    def execute_preprocessing(self, dataset):

        # MinMax normalization of dataset 
        for i in dataset.columns:
            dataset[i] = (dataset[i] - dataset[i].min()) / (dataset[i].max() - dataset[i].min())
        
        # substitute zero by 1*10^-6 to avoid division by zero in preprocessing
        dataset[dataset == 0] = 0.000001

        # execute preprocessing based on each strategy
        if self.mode == 'angle':
            output_df = np.apply_along_axis(self.angle, 1, dataset.values)

        elif self.mode == 'radius':
            output_df = np.apply_along_axis(self.radius, 1, dataset.values)

        elif self.mode == 'angleradius':
            output_df = np.apply_along_axis(self.angleradius, 1, dataset.values)

        else:
            print("ERROR: mode not found, choose 'angle', 'radius' or 'angleradius' ")
        
        return pd.DataFrame(output_df)