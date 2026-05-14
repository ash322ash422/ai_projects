import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Title and description
st.title("Bank Customer Churn Prediction")
st.write("This application predicts whether a customer is likely to churn based on their banking data.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(data.head())

    # Display basic stats and data info
    st.write("### Dataset Information:")
    st.write(data.describe(include='all'))
    st.write("### Columns:", list(data.columns))

    # Selecting features and target
    st.write("### Select Features and Target")
    all_columns = data.columns.tolist()
    target_column = st.selectbox("Select the Target Column", all_columns, index=all_columns.index('Exited'))
    feature_columns = st.multiselect("Select Feature Columns", [col for col in all_columns if col != target_column], default=[
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ])

    if target_column and feature_columns:
        X = pd.get_dummies(data[feature_columns], drop_first=True)
        y = data[target_column]

        # Splitting the data
        st.write("### Splitting the Dataset")
        test_size = st.slider("Test Size (in %)", min_value=10, max_value=50, value=30, step=5) / 100.0
        random_state = st.number_input("Random State (for reproducibility)", value=42, step=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.write(f"Train Dataset: {X_train.shape}, Test Dataset: {X_test.shape}")

        # Apply SMOTE to handle class imbalance
        st.write("### Handling Class Imbalance with SMOTE")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.write(f"Resampled Train Dataset: {X_train.shape}, Resampled Labels: {y_train.value_counts().to_dict()}")

        # Model selection
        st.write("### Model Selection")
        model_options = {
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Support Vector Machine': SVC()
        }
        selected_model_name = st.selectbox("Select a Model", list(model_options.keys()))
        selected_model = model_options[selected_model_name]

        # Hyperparameter tuning
        st.write("### Hyperparameter Tuning")
        if selected_model_name == 'Random Forest':
            n_estimators = st.slider("Number of Estimators (Trees)", min_value=10, max_value=200, value=100, step=10)
            max_depth = st.slider("Max Depth of Tree", min_value=1, max_value=20, value=10, step=1)
            selected_model.set_params(n_estimators=n_estimators, max_depth=max_depth)
        elif selected_model_name == 'Gradient Boosting':
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100, step=10)
            selected_model.set_params(learning_rate=learning_rate, n_estimators=n_estimators)
        elif selected_model_name == 'Support Vector Machine':
            kernel = st.selectbox("Kernel Type", ['linear', 'poly', 'rbf', 'sigmoid'])
            C = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            selected_model.set_params(kernel=kernel, C=C)

        # Train the selected model
        st.write("### Train the Model")
        selected_model.fit(X_train, y_train)

        # Make predictions
        y_pred = selected_model.predict(X_test)

        # Model evaluation
        st.write("### Model Evaluation")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.write("Classification Report:")
        cr = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(cr).transpose())

        # Feature importance (if applicable)
        if hasattr(selected_model, 'feature_importances_'):
            st.write("### Feature Importance")
            feature_importance = pd.Series(selected_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(feature_importance)

        # Prediction on user input
        st.write("### Predict on New Data")
        new_data = {}
        for col in feature_columns:
            if data[col].dtype == 'object':
                unique_vals = data[col].unique()
                new_data[col] = st.selectbox(f"Select value for {col}", options=unique_vals)
            else:
                new_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

        if st.button("Predict Churn"):
            input_df = pd.get_dummies(pd.DataFrame([new_data]), drop_first=True)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            prediction = selected_model.predict(input_df)[0]
            st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
