# Instructions:
- I used python 3.11
- install: `pip install streamlit==1.41.1 joblib==1.4.2 scikit-learn==1.6.1`
- to run: `streamlit run app.py`

## Deploy model on streamlit cloud:
1. Goto https://streamlit.io/ and sign up.
2. After filling few basic questionairre, you will be taken to screen where you will see on RHS Top a button 'Create APP' . Click on it.
3. Then it will inform you that you must be connected to github to deploy your app. Click on 'Connect to github'
4. You will be redirected to github site where you grant permission to 'Authorize streamlit' to access your github account. You will be redireced to streamlit page
5. On streamlit page you will receive 2-3 options on how do you want to deploy your app. Choose 'Deploy public app from github'
6. You will be prompted to enter following information:
   - repository: ash322ash422/ai_projects
   - Branch: master
   - main filepath name: tutorial/tut_ml/bigmart_sales_prediction/app.py
   - app url(optional): aiprojects-tutorial 

Click Deploy. You will receive message 'you app is in the oven'.