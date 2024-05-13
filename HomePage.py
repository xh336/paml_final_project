import streamlit as st                  # pip install streamlit

st.set_page_config(
    page_title="Home Page",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded",
    
)
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Standardized Testing Score Prediction")
with col2:
    st.image("test.jpg", width=200)


#############################################

st.markdown("This application utilizes Machine Learning Models to create Standardized Testing Score Predictions! \
            With this model, you are able to input demographic information to predict \
            the score for a Standardized Test in math, reading, or writing given to high school students.")

st.markdown("### Introduction")

st.markdown("""This project aims to explore the complex factors affecting student academic performance \
            by developing a predictive model based on demographic variables. The significance of this \
            study lies in its potential to enhance educational equity and access by identifying key \
            predictors of academic outcomes. This can help in formulating targeted interventions for \
            students needing additional support. Utilizing machine learning techniques, such as regression \
            and classification, the model will not only handle complex variable interactions but also offer \
            interpretable insights, focusing particularly on demographic influences to reveal subtle but \
            impactful relationships in student performance data.""")

st.markdown("### Instructions")

st.markdown("""
            
The example dataset is a simulated dataset updated in 2024 on Kaggle. It includes all eight \
input features of 1,000 sand their standardized test scores (writing, reading, math scores. 

**The user would input these features:** 
- Gender: Male or Female
- Race/Ethnicity: A, B, C or D
- Lunch: Free/reduced or Standard
- Test preparation course: None or Completed
- Parental level of education: Some High School, High School, Some College, Associate's Degree, Bachelor's Dgree'r's Degree
- re:  Score 0 :re:M:, Score:g Score: 0 - 100, :0
- SAT Writing Score: 0 - 100
""")

st.markdown(""" **This web application will allow you to:** 

1. Explore and visualize the data to gain insights.
2. Preprocess and prepare the data for machine learning algorithms.
3. Select a model and train it. 
4. Test Model. Evaluate models using standard metrics.
5. Deploy your application.
""")

st.markdown(""" 
""")

st.markdown("Click **Explore and Preprocess Dataset** to get started.")