import streamlit as st                  # pip install streamlit

st.markdown("# Maching Learning Model for SAT Score Prediction")

#############################################

st.markdown("### Practical Applications of Machine Learning (PAML) Final Project")

#############################################

st.markdown("Welcome to the Maching Learning Model for SAT Score Prediction! \
            With this model, you would be able to use demographic information input to predict \
            the SAT score of students.")

st.markdown("### Introduction")

st.markdown("""The motivation behind this project is to investigate the multifaceted dynamics \
            influencing students' academic performance and to develop a predictive model that \
            can effectively estimate students' scores based on demographic factors.  This problem \
            is particularly important due to its implications for educational equity and access, \
            as understanding the factors contributing to differential academic outcomes can inform \
            targeted interventions to support marginalized student populations. The application \
            developed in this project aims to provide insights into the predictors of student \
            performance and offer a tool for educators and policymakers to identify students who \
            may benefit from additional support or resources. \
            
            Previous studies have shown contentious results on the correlations between demographic \
            factors and academic performances, and extensive models have been generated to predict \
            study performances with significant disparities among different student groups. However, \
            with focus on different demographic factors or research aims, there was not only \
            substantial work on solving problems or completing assignments but even on improving student \
            learning outcomes. \
            
            From a technical standpoint, the proposed approach leverages machine learning methods \
            such as regression analysis or classification algorithms to model the relationship between \
            demographic variables and student scores. These methods are well-suited to address the \
            problem as they can handle complex interactions between multiple predictor variables and \
            produce interpretable results. What sets this approach apart is its focus on demographic \
            factors specifically and its potential to uncover nuanced relationships that may not be \
            apparent through traditional statistical methods alone. 
""")

st.markdown("### Instructions")

st.markdown("""
            
The example dataset is a simulated dataset updated in 2024 on Kaggle. It includes all eight \
input features of 1,000 students and their corresponding SAT scores. 

**The user would input these features:** 
- Gender: Male or Female
- Race/Ethnicity: A, B, C or D
- Lunch: Free/reduced or Standard
- Test preparation course: None or Completed
- Parental level of education: Some High School, High School, Some College, Associate's Degree, Bachelor's Degree, Master's Degree

**The output features are:**
- SAT Math Score: 0 - 100
- SAT Reading Score: 0 - 100
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