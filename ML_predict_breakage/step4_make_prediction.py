from step3_train_and_save_models import load_model_and_encoder
from config import ( MODEL_DECISION_TREE, MODEL_LOGISTICS_REGRESSION,
                    MODEL_SGDClassifier, MODEL_SVC, SAVED_DIR
)

if __name__ == '__main__':
    print("################DecisionTreeClassifier####################")
    model = MODEL_DECISION_TREE
    saved_model_path = SAVED_DIR / (model + '.pkl')
    loaded_model,loaded_encoder  = load_model_and_encoder(saved_model_path)
    
    new_data = [(75,'Crizol','plastic','Cathy','-3 to 0')]
    
    X_encoded = loaded_encoder.transform(new_data) # X_encoded= [[1. 0. 0. 2. 1.]]
    print("X_encoded=",X_encoded)
    
    y_predict = loaded_model.predict(X_encoded)
    y_predict_proba = loaded_model.predict_proba(X_encoded)
    y_predict_log_proba = -loaded_model.predict_log_proba(X_encoded)
    y_feature_importance = loaded_model.feature_importances_
    print("y_predict=",y_predict,
          "\ny_predict_proba=",y_predict_proba,
          "\ny_predict_log_proba",y_predict_log_proba,
          "\ny_feature_importance=",y_feature_importance
    )
    
    print("################LogisticRegression####################")
    model = MODEL_LOGISTICS_REGRESSION
    saved_model_path = SAVED_DIR / (model + '.pkl')
    loaded_model,loaded_encoder  = load_model_and_encoder(saved_model_path)
    
    new_data = [(75,'Crizol','plastic','Cathy','-3 to 0')]
    
    X_encoded = loaded_encoder.transform(new_data) # X_encoded= [[1. 0. 0. 2. 1.]]
    print("X_encoded=",X_encoded)
    
    y_predict = loaded_model.predict(X_encoded)
    y_predict_proba = loaded_model.predict_proba(X_encoded)
    y_predict_log_proba = -loaded_model.predict_log_proba(X_encoded)
#     y_feature_importance = loaded_model.feature_importances_ # does not exist
    print("y_predict=",y_predict,
          "\ny_predict_proba=",y_predict_proba,
          "\ny_predict_log_proba",y_predict_log_proba,
      #     "\ny_feature_importance=",y_feature_importance
    )
    
    print("################SGDClassifier####################")
    model = MODEL_SGDClassifier
    saved_model_path = SAVED_DIR / (model + '.pkl')
    loaded_model,loaded_encoder  = load_model_and_encoder(saved_model_path)
    
    new_data = [[75,'Crizol','plastic','Cathy','-3 to 0'],]
    
    X_encoded = loaded_encoder.transform(new_data) # X_encoded= [[1. 0. 0. 2. 1.]]
    print("X_encoded=",X_encoded)
    
    y_predict = loaded_model.predict(X_encoded)
    y_predict_proba = loaded_model.predict_proba(X_encoded)
    y_predict_log_proba = -loaded_model.predict_log_proba(X_encoded)
#     y_feature_importance = loaded_model.feature_importances_ # does not exist
    print("y_predict=",y_predict,
          "\ny_predict_proba=",y_predict_proba,
          "\ny_predict_log_proba",y_predict_log_proba,
      #     "\ny_feature_importance=",y_feature_importance
    )

    print("################SVC####################")
    model = MODEL_SVC
    saved_model_path = SAVED_DIR / (model + '.pkl')
    loaded_model,loaded_encoder  = load_model_and_encoder(saved_model_path)
    
    new_data = [[75,'Crizol','plastic','Cathy','-3 to 0'],]
    
    X_encoded = loaded_encoder.transform(new_data) # X_encoded= [[1. 0. 0. 2. 1.]]
    print("X_encoded=",X_encoded)
    
    y_predict = loaded_model.predict(X_encoded)
    y_predict_proba = loaded_model.predict_proba(X_encoded)
    y_predict_log_proba = -loaded_model.predict_log_proba(X_encoded)
#     y_feature_importance = loaded_model.feature_importances_ # does not exist
    print("y_predict=",y_predict,
          "\ny_predict_proba=",y_predict_proba,
          "\ny_predict_log_proba",y_predict_log_proba,
      #     "\ny_feature_importance=",y_feature_importance
    )
