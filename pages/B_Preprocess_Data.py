import streamlit as st
import pandas as pd

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 0 - Introduction to Streamlit")

#############################################

st.markdown('# Preprocess Dataset')

###################### FETCH or RESTORE DATASET #######################

dataset = st.file_uploader("Upload a Dataset")
if (dataset):
  dataset_df = pd.read_csv(dataset)
  if 'dataset_key' not in st.session_state:
    st.session_state.dataset_key = dataset_df



# Restore dataset if already in memory
if 'dataset_key' in st.session_state:
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    
    ds_df = st.session_state.dataset_key

    # Display dataframe as table
    st.dataframe(ds_df)

    # Show summary of missing values including 
    #   1) number of categories with missing values,
    isna_num = ds_df.isna().any().sum()
    st.markdown("Number of categories with missing values: " + str("%.2f" % round(isna_num, 2)))

    #   2) average number of missing values per category
    isna_ave = ds_df.isna().sum().sum()/10
    st.markdown("Average number of missing values per category: " + str("%.2f" % round(isna_ave, 2)))

    #   3) Total number of missing values
    isna_sum = ds_df.isna().sum().sum()
    st.markdown("Total number of missing numbers: " + str("%.2f" % round(isna_sum, 2)))

        ############################################# MAIN BODY #############################################
        
        # Descriptive Statistics 
    st.markdown("### Summary of Descriptive Statistics")
    columns = list(ds_df.select_dtypes(['float','int']).columns)
    multi_feature = st.multiselect("Select features for statistics", columns)

        # Compute Descriptive Statistics including mean, median, min, max
    multi_stat = st.multiselect("Select statistics to display", ("Min", "Max", "Median", "Mean"))
    def feature_disp(featurelist, statlist):
        for i in featurelist:
            stat_string = ""
            for j in statlist:
                if j == "Min":
                    stat_string += ("min: " + str(round(ds_df[i].min(), 2)) + " | ")
                elif j == "Max":
                    stat_string += ("max: " + str(round(ds_df[i].max(), 2)) + " | ")
                elif j == "Median":
                    stat_string += ("median: " + str(round(ds_df[i].median(), 2)) + " | ")
                else:
                    stat_string += ("mean: " + str(round(ds_df[i].mean(), 2)) + " | ")
            st.write(i, stat_string)

    display = feature_disp(multi_feature, multi_stat)
