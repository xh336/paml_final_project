import streamlit as st                  
import pandas as pd
import plotly.express as px

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 0 - Introduction to Streamlit")

#############################################

st.markdown('# Explore Dataset')

###################### FETCH DATASET #######################

# Dataset upload
st.markdown("### Import Dataset")
dataset = st.file_uploader("Upload a Dataset")

if (dataset):
  dataset_df = pd.read_csv(dataset)
  if 'dataset_key' not in st.session_state:
    st.session_state.dataset_key = dataset_df

###################### EXPLORE DATASET #######################

# Restore dataset if already in memory
if 'dataset_key' in st.session_state:
  ds_df = st.session_state.dataset_key

  # Display feature names and descriptions 
  st.markdown("#### Explore Dataset Feasures")
  st.markdown("Feature 0 - longitude")
  st.markdown("Feature 1 - latitude")
  st.markdown("Feature 2 - housing_median_age")
  st.markdown("Feature 3 - total_rooms")
  st.markdown("Feature 4 - total_bedrooms")
  st.markdown("Feature 5 - population")
  st.markdown("Feature 6 - households")
  st.markdown("Feature 7 - median_income")
  st.markdown("Feature 8 - median_house_value")
  st.markdown("Feature 9 - ocean_proximity")

  # Display dataframe as table
  st.dataframe(ds_df)

  #X = df

  ###################### VISUALIZE DATASET #######################

  st.sidebar.header("Create Histogram Plots")
  st.sidebar.header("Specify Input Parameters")

  # Collect user plot selection
  numeric_columns = list(ds_df.select_dtypes(['float','int']).columns)
  selected_feature = st.sidebar.selectbox("X axis", numeric_columns)

  # Specify Input Parameters
  def sidebar_filter(df, values):
    """
    Input: df is pandas dataframe containing dataset; values is a list of value for the filter
    Output: dictionary of sidebar filters on features
    """
    side_bar_data = {}
    try:
        f = st.sidebar.slider(str(values), float(df[str(values)].min()), 
                              float(df[str(values)].max()), float(df[str(values)].mean()))
        side_bar_data[values] = f
    except Exception as e:
        print(e)
    return side_bar_data
  
  # def sidebar_filter(feature, v1, v2):
  #   feature_min = float(ds_df[selected_feature].min())
  #   feature_max = float(ds_df[selected_feature].max())
  #   min_selected, max_selected = st.sidebar.slider(selected_feature, feature_min, 
  #                                                 feature_max, (v1, v2))
  #   return min_selected, max_selected

  # filter_min, filter_max = sidebar_filter(selected_feature, ds_df[selected_feature].min(), ds_df[selected_feature].max())
  # filter_df = ds_df[(ds_df[selected_feature] >= filter_min) & (ds_df[selected_feature] <= filter_max)]
  
  selected_data = sidebar_filter(ds_df, selected_feature)
  filter_df = ds_df[ds_df[selected_feature] <= selected_data[selected_feature]]
  # st.dataframe(filter_df)

  # Plot Histogram
  st.markdown("#### Visualize Feasures")
  feature_plot = px.histogram(filter_df, selected_feature)
  st.plotly_chart(feature_plot)

  st.markdown("Continue to data preprocessing.")