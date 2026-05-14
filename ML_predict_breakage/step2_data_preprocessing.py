from libraries import *
from sklearn.preprocessing import OrdinalEncoder
from step1_data_collection import collect_data

def preprocess_data(df):
    
    # Define custom bins and labels for OD_SPH
    # Use pd.cut() to bin OD_SPH values
    bins = [-6, -3, 0, 3, 6]
    labels = ['-6 to -3', '-3 to 0', '0 to +3', '+3 to +6']
    df['OD_SPH_binned'] = pd.cut(df['OD_SPH'], bins=bins, labels=labels)
    df = df.dropna(subset=['OD_SPH_binned'])
    
    encoder = OrdinalEncoder()
    categorical_columns = ["DIA", "MANF", "MATERIAL", "OPERATOR", "OD_SPH_binned"]
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    
    X = df[categorical_columns]
    y = np.array(df['QUALITY'])
    return X,y,encoder
    
    
if __name__ == '__main__':
    df = collect_data()
    X,y,encoder = preprocess_data(df)
    print("X=\n",X)
    print()
    print("y=",y)
    
