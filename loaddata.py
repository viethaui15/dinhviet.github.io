import pandas as pd

def load_data():
    df = pd.read_csv('DATA\Mall_Customers.csv')
    df['Gender'] = df['Genre']
    df = df.drop('Genre', axis=1)
    df = df[['CustomerID','Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    X = df[['Annual Income (k$)', 'Spending Score (1-100)',]].values
    gender_mapping = {'Male': 1, 'Female': 0}
    df['Gender'] = df['Gender'].map(gender_mapping)
    return X