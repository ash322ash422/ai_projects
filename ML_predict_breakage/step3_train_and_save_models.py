from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from  pathlib import Path
import pickle

from step1_data_collection import collect_data
from step2_data_preprocessing import preprocess_data

from config import (MODEL_DECISION_TREE, 
                    MODEL_LOGISTICS_REGRESSION,
                    MODEL_SGDClassifier,
                    MODEL_SVC, 
                    SAVED_DIR
)
RANDOM_STATE = 42

def train_and_score_model(X, y, model: str):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)
    
    if model == MODEL_DECISION_TREE:
        classifier = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    elif model == MODEL_LOGISTICS_REGRESSION:
        classifier = LogisticRegression(random_state=RANDOM_STATE)
    elif model == MODEL_SGDClassifier:
        classifier = SGDClassifier(loss="modified_huber",random_state=RANDOM_STATE)
    elif model == MODEL_SVC:
        classifier = SVC(probability=True,random_state=RANDOM_STATE)
    else:
        classifier = None    
    
    classifier.fit(X,y)
    score = classifier.score(X,y)
    return classifier, score
    
def save_model_and_encoder(model, model_path,encoder): 
    model_data = {"model":model, "encoder":encoder}
    with open(model_path, 'wb') as file:
        pickle.dump(model_data, file)

def load_model_and_encoder(model_path: str): 
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    
    return loaded_model["model"], loaded_model["encoder"]

if __name__ == '__main__':
    df = collect_data()
    X,y,encoder = preprocess_data(df)
    
    ################DecisionTreeClassifier####################
    model = MODEL_DECISION_TREE
    saved_model_path = SAVED_DIR / (model + '.pkl')
    train_model, train_score = train_and_score_model(X,y,model=model)
    print("DecisionTreeClassifier train_score=",train_score)
    save_model_and_encoder(train_model, saved_model_path,encoder=encoder)
    loaded_model,loaded_encoder = load_model_and_encoder(saved_model_path)
    
    ################LogisticRegression####################
    model = MODEL_LOGISTICS_REGRESSION
    saved_model_path = SAVED_DIR / (model + '.pkl')
    train_model, train_score = train_and_score_model(X,y,model=model)
    print("LogisticRegression train_score=",train_score)
    save_model_and_encoder(train_model, saved_model_path,encoder=encoder)
    loaded_model,loaded_encoder = load_model_and_encoder(saved_model_path)
    
    ################SGDClassifier####################
    model = MODEL_SGDClassifier
    saved_model_path = SAVED_DIR / (model + '.pkl')
    train_model, train_score = train_and_score_model(X,y,model=model)
    print("SGDClassifier train_score=",train_score)
    save_model_and_encoder(train_model, saved_model_path,encoder=encoder)
    loaded_model,loaded_encoder = load_model_and_encoder(saved_model_path)
    
    ################SVC####################
    model = MODEL_SVC
    saved_model_path = SAVED_DIR / (model + '.pkl')
    train_model, train_score = train_and_score_model(X,y,model=model)
    print("SVC train_score=",train_score)
    save_model_and_encoder(train_model, saved_model_path,encoder=encoder)
    loaded_model,loaded_encoder = load_model_and_encoder(saved_model_path)
    