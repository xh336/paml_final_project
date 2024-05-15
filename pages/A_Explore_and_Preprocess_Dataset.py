import streamlit as st                  
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

feature_lookup = {
    'gender':'**gender** - female, male',
    'race_ethnicity':'**race_ethnicity** - group A, B, C, D, E',
    'parental_level_of_education':'**parental_level_of_education** - some high school, high school, some college, associate\'s degree, bachelor\'s degree, master\'s degree',
    'lunch':'**total_rooms** - standard, free/reduced',
    'test_preparation_course':'**test_preparation_course** - none, completed'
}

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')

def load_dataset(filepath):
    data=None
    data = pd.read_csv(filepath)
    st.session_state['house_df'] = data
    return data
def display_features(df,feature_lookup):
    for idx, col in enumerate(df.columns):
        if col in feature_lookup:
            st.markdown('Feature %d - %s'%(idx, feature_lookup[col]))
        else:
            st.markdown('Feature %d - %s'%(idx, col))
def sidebar_filter(df, chart_type, x=None, y=None):
    side_bar_data = []
    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)

    return side_bar_data
def summarize_missing_data(df, top_n=3):
    out_dict = {'num_categories': 0,
                'average_per_category': 0,
                'total_missing_values': 0,
                'top_missing_categories': []}
    
    missing_column_counts = df[df.columns[df.isnull().any()]].isnull().sum()
    max_idxs = np.argsort(missing_column_counts.to_numpy())[::-1][:top_n]

    out_dict['num_categories'] = df.isna().any(axis=0).sum()
    out_dict['average_per_category'] = df.isna().sum().sum()/len(df.columns)
    out_dict['total_missing_values'] = df.isna().sum().sum()
    out_dict['top_missing_categories'] = df.columns[max_idxs[:top_n]].to_numpy()
    return out_dict
def remove_features(df,removed_features):
    df = df.drop(columns=removed_features)
    st.session_state['house_df'] = df
    return df
def one_hot_encode_feature(df, feature):
    one_hot_cols = pd.get_dummies(df[[feature]], prefix=feature, prefix_sep='_',dummy_na=True)
    df = pd.concat([df, one_hot_cols], axis=1)
    st.session_state['house_df'] = df
    return df
def integer_encode_feature(df, feature):
    enc = OrdinalEncoder()
    df.insert(len(df.columns), feature+'_int', enc.fit_transform(df[[feature]]))   
    #remove the original column
    #df.drop(feature,axis=1,inplace=True)
    st.session_state['house_df'] = df
    return df
def scale_features(df, features, scaling_method): 
    if (scaling_method == 'Standardarization'):
        for f in features:
            df[f+'_std']=(df[f]-df[f].mean())/(df[f].std())
            #st.write('STD', df[f+'_std'].std())
            #st.write('mean', df[f+'_std'].mean())
    elif (scaling_method == 'Normalization'):
        for f in features:
            df[f+'_norm']=(df[f]-df[f].min())/(df[f].max()-df[f].min())
            #st.write('min', df[f+'_norm'].min())
            #st.write('max', df[f+'_norm'].max())
    else:
        for f in features:
            #test_homework1.py uses Base-2 logarithm
            df[f+'_log']=np.log2(df[f]+0.0000001)            
    #st.write(df)
    st.session_state['house_df'] = df
    return df
def create_feature(df, math_select, math_feature_select, new_feature_name):
    df.dropna()
    if math_select == "square root":
        df[new_feature_name] = np.sqrt(df[math_feature_select])
    elif math_select == "add":
       df[new_feature_name] = df[math_feature_select].sum(axis = 1)
    elif math_select == "subtract":
        df[new_feature_name] = df[math_feature_select[0]].sub(df[math_feature_select[1]])
    elif math_select == "multiply":
        df[new_feature_name] = df[math_feature_select[0]] * df[math_feature_select[1]]
    elif math_select == "divide":
        df[new_feature_name] = df[math_feature_select[0]] / df[math_feature_select[1]]
    elif math_select == "ceil":
        df[new_feature_name] = np.ceil(df[math_feature_select])
    else:
        df[new_feature_name] = np.floor(df[math_feature_select])
    st.session_state['house_df'] = df
    return df
def remove_outliers(df, feature, outlier_removal_method=None):
    df = df.dropna()
    if outlier_removal_method == "IQR":
        q1 = np.percentile(df[feature], 25, axis = 0)
        q3 = np.percentile(df[feature], 75, axis = 0)
        IQR = q3 - q1
        upper_bound = q3 + 1.5 * IQR
        lower_bound = q1 - 1.5 * IQR
        df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
        df.dropna()
    else:
        lower_bound = df[feature].mean() - 3 * df[feature].std()
        upper_bound = df[feature].mean() + 3 * df[feature].std()
        df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
        df.dropna()
    st.session_state['house_df'] = df
    return df, lower_bound, upper_bound
def compute_descriptive_stats(df, stats_feature_select, stats_select):
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    for feature in stats_feature_select:
        for stat in stats_select:
            if stat == "Mean":
                out_dict["mean"] = round(df[feature].mean(), 2)
                output_str = output_str + " " + feature + ' mean: %.2f |' % (out_dict["mean"])
            elif stat == "Median":
                df = df.dropna()
                out_dict["median"] = np.median(df[feature])
                output_str = output_str + " " + feature + ' median: %.2f |' % (out_dict["median"])
            elif stat == "Max":
                out_dict["max"] = df[feature].max()
                output_str = output_str + " " + feature + ' max: %.2f |' % (out_dict["max"])
            else:
                out_dict["min"] = df[feature].min()
                output_str = output_str + " " + feature + ' min: %.2f |' % (out_dict["min"])
    st.markdown(output_str)
    return output_str, out_dict
def compute_correlation(df, features):
    correlation = df[features].corr()
    pairs = combinations(features, 2)
    cor_summary_statements = []
    for f1, f2 in pairs:
        cor = correlation[f1][f2]
        summary = '- Features %s and %s are %s %s correlated: %.2f' % (
            f1, f2, 'strongly' if cor > 0.5 else 'weakly', 'positively' if cor > 0 else 'negatively', cor)
        cor_summary_statements.append(summary)
        st.markdown(summary)
    return correlation, cor_summary_statements
###################### FETCH DATASET #######################
df=None

filename = 'datasets/study_performance.csv'
if('house_df' in st.session_state):
    df = st.session_state['house_df']
else:
    if(filename):
        df = load_dataset(filename)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.markdown('### 1. Explore Dataset Features')

    # Display feature names and descriptions (from feature_lookup)
    display_features(df,feature_lookup)
    
    # Display dataframe as table
    st.dataframe(df)

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Visualize Numeric Features')

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    #st.write(numeric_columns)

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots','Lineplots','Histogram','Boxplot']
    )

    side_bar_data=[]

    # Draw plots
    if chart_select == 'Scatterplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.scatter(data_frame=df,
                                x=x_values, y=y_values,
                                range_x=[side_bar_data[0][0],
                                        side_bar_data[0][1]],
                                range_y=[side_bar_data[1][0],
                                        side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df,
                                x=x_values,
                                range_x=[side_bar_data[0][0],
                                            side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Lineplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.line(df,
                            x=x_values,
                            y=y_values,
                            range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]],
                            range_y=[side_bar_data[1][0],
                                    side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.box(df,
                            x=x_values,
                            range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)

    if(side_bar_data):
        if st.sidebar.button('Clip feature from %.2f to %.2f' % (side_bar_data[0][0], side_bar_data[0][1])):
            df[x_values+'_clipped'] = df[x_values]
            df[df[x_values+'_clipped']>side_bar_data[0][1]] = 0
            df[df[x_values+'_clipped']<side_bar_data[0][0]] = 0
            st.sidebar.write(x_values + ' cliped from '+str(side_bar_data[0][0])+' to '+str(side_bar_data[0][1]))
            if(chart_select == 'Scatterplots' or chart_select == 'Lineplots'):
                df[y_values+'_clipped'] = df[y_values]
                df[df[y_values+'_clipped']>side_bar_data[1][1]] = 0
                df[df[y_values+'_clipped']<side_bar_data[1][0]] = 0
                st.sidebar.write(y_values + ' cliped from '+str(side_bar_data[1][0])+' to '+str(side_bar_data[1][1]))

    # Display original dataframe
    st.markdown('## 3. View initial data with missing values or invalid inputs')
    st.dataframe(df)

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Show summary of missing values including 
    missing_data_summary = summarize_missing_data(df)

    st.markdown("Number of categories with missing values: " + str("%.2f" % round(missing_data_summary['num_categories'], 2)))
    st.markdown("Average number of missing values per category: " + str("%.2f" % round(missing_data_summary['average_per_category'], 2)))
    st.markdown("Total number of missing values: " + str("%.2f" % round(missing_data_summary['total_missing_values'], 2)))
    

    # Remove param
    st.markdown('### 4. Remove irrelevant/useless features')
    removed_features = st.multiselect(
        'Select features',
        df.columns,
    )
    df = remove_features(df, removed_features)

    ########
    # Display updated dataframe
    st.dataframe(df)

    ############################################# PREPROCESS DATA #############################################

    # Handling Text and Categorical Attributes
    st.markdown('### 5. Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with (int_col):
        text_feature_select_int = st.selectbox(
            'Select text features for Integer encoding',
            string_columns,
        )
        if (text_feature_select_int and st.button('Integer Encode feature')):
            df = integer_encode_feature(df, text_feature_select_int)
    
    # Perform One-hot Encoding
    with (one_hot_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for One-hot encoding',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('One-hot Encode feature')):
            df = one_hot_encode_feature(df, text_feature_select_onehot)

    # Show updated dataset
    st.write(df)

    # Sacling features
    st.markdown('### 6. Feature Scaling')
    st.markdown('Use standardarization or normalization to scale features')

    # Use selectbox to provide impute options {'Standardarization', 'Normalization', 'Log'}
    scaling_method = st.selectbox(
        'Select feature scaling method',
        ('Standardarization', 'Normalization', 'Log')
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    scale_features_select = st.multiselect(
        'Select features to scale',
        numeric_columns,
    )

    # Call scale_features function to scale features
    df = scale_features(df, scale_features_select, scaling_method)
    #########

    # Display updated dataframe
    st.dataframe(df)

    # Create New Features
    st.markdown('## 7. Create New Features')
    st.markdown(
        'Create new features by selecting two features below and selecting a mathematical operator to combine them.')
    math_select = st.selectbox(
        'Select a mathematical operation',
        ['add', 'subtract', 'multiply', 'divide', 'square root', 'ceil', 'floor'],
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    if (math_select):
        if (math_select == 'square root' or math_select == 'ceil' or math_select == 'floor'):
            math_feature_select = st.multiselect(
                'Select features for feature creation',
                numeric_columns,
            )
            sqrt = np.sqrt(df[math_feature_select])
            if (math_feature_select):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    if (new_feature_name):
                        df = create_feature(
                            df, math_select, math_feature_select, new_feature_name)
                        st.write(df)
        else:
            math_feature_select1 = st.selectbox(
                'Select feature 1 for feature creation',
                numeric_columns,
            )
            math_feature_select2 = st.selectbox(
                'Select feature 2 for feature creation',
                numeric_columns,
            )
            if (math_feature_select1 and math_feature_select2):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    df = create_feature(df, math_select, [
                                        math_feature_select1, math_feature_select2], new_feature_name)
                    st.write(df)

    st.markdown('### 8. Inspect Features for outliers')
    outlier_feature_select = None
    numeric_columns = list(df.select_dtypes(include='number').columns)

    outlier_method_select = st.selectbox(
        'Select statistics to display',
        ['IQR', 'STD']
    )

    outlier_feature_select = st.selectbox(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        df, lower_bound, upper_bound = remove_outliers(
            df, outlier_feature_select, outlier_method_select)
        st.write('Outliers for feature %s are lower than %.2f and higher than %.2f' % (
            outlier_feature_select, lower_bound, upper_bound))
        st.write(df)

    # Descriptive Statistics 
    st.markdown('### 9. Summary of Descriptive Statistics')

    stats_numeric_columns = list(df.select_dtypes(['float','int']).columns)
    stats_feature_select = st.multiselect(
        'Select features for statistics',
        stats_numeric_columns,
    )

    stats_select = st.multiselect(
        'Select statistics to display',
        ['Mean', 'Median','Max','Min']
    )
            
    # Compute Descriptive Statistics including mean, median, min, max
    display_stats, display_dict = compute_descriptive_stats(df, stats_feature_select, stats_select)

    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### 10. Correlation Analysis")

    # Collect features for correlation analysis using multiselect
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )

    # Compute correlation between selected features
    correlation, correlation_summary = compute_correlation(
        df, select_features_for_correlation)
    st.write(correlation)

    # Display correlation of all feature pairs
    if select_features_for_correlation:
        try:
            fig = scatter_matrix(
                df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)