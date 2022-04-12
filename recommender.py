import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle


dataset = pd.read_csv('dataset.csv')



class Recommender:
    
    def __init__(self):
        self.df = pd.read_csv('dataset.csv')
    
    def get_features(self):
        nutrient_dummies = self.df.Nutrient.str.get_dummies()
        disease_dummies = self.df.Disease.str.get_dummies(sep=' ')
        diet_dummies = self.df.Diet.str.get_dummies(sep=' ')
        feature_df = pd.concat([nutrient_dummies,disease_dummies,diet_dummies],axis=1)

        return feature_df
    

    def k_neighbor(self,inputs):
        
        feature_df = self.get_features()
        
        #initializing model with k=20 neighbors
        model = NearestNeighbors(n_neighbors=40,algorithm='ball_tree')
        
        # fitting model with dataset features
        model.fit(feature_df)
        #save model as pickLE from pickle library
        pickle.dump(model,open('model.pkl','wb'))

        # get results 
        #df_results = get_recommendation(inputs)
        #return df_results


  