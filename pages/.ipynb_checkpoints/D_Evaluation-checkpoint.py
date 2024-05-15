import numpy as np                  
import pandas as pd                  
import streamlit as st                 
from pages.B_Train_Base_Model import split_dataset
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
import plotly.express as px

st.title('Model Testing')

#############################################
def rmse(y_true, y_pred):
    error=0
    error = np.sqrt(np.sum(np.power(y_pred-y_true,2))/len(y_true))
    return error
def mae(y_true, y_pred):
    error=0
    error = (np.sum(np.abs(y_pred-y_true))) / len(y_true)
    return error
def r2(y_true, y_pred):
    error=0
    tss = np.sum(np.power(y_true - np.mean(y_pred),2))
    rss = np.sum(np.power(y_true - y_pred,2))
    error = 1 - (rss/tss)
    return error
METRICS_MAP = {
    'mean_absolute_error': mae,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

# Helper function
def compute_eval_metrics(X, y_true, model, metrics):
    metric_dict = {}
    y_pred = model.predict(X)
    for metric in metrics:
        metric_dict[metric] = METRICS_MAP[metric](y_true, y_pred)
    return metric_dict


def plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metrics, model_name):
    fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()
    for i, metric in enumerate(metrics):
        metric_fn = METRICS_MAP[metric]
        train_errors, val_errors = [], []
        for m in range(500, len(X_train)+1, 500):
            trained_model.fit(X_train[:m], y_train[:m])
            y_train_predict = trained_model.predict(X_train[:m])
            y_val_predict = trained_model.predict(X_val)
            train_errors.append(metric_fn(y_train[:m], y_train_predict))
            val_errors.append(metric_fn(y_val, y_val_predict))

        fig.add_trace(go.Scatter(x=np.arange(500, len(X_train)+1, 500),
                      y=train_errors, name=metric_fn.__name__+" Train"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(500, len(X_train)+1, 500),
                      y=val_errors, name=metric_fn.__name__+" Val"), row=i+1, col=1)

        fig.update_xaxes(title_text="Training Set Size")
        fig.update_yaxes(title_text=metric_fn.__name__, row=i+1, col=1)
        fig.update_layout(title=model_name)

        df[metric_fn.__name__ + " Training Set"] = train_errors
        df[metric_fn.__name__ + " Validation Set"] = val_errors
    return fig, df
def perform_cross_validation(X,y,model,n_folds,metrics):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
    cv_results = {metric: [] for metric in metrics}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for metric in metrics:
            score = METRICS_MAP[metric](y_test, y_pred)
            cv_results[metric].append(score)
    cv_results = {metric: np.mean(scores) for metric, scores in cv_results.items()}
    return cv_results
def plot_residuals(y_true, y_pred, model_name):
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    residuals = y_true -  y_pred
    fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                     title=f'Residual Plot for {model_name}')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title='Predicted Values',
                      yaxis_title='Residuals',
                      title=f'Residuals Plot for {model_name}')
    return fig
    

def restore_data(df):
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if('target' in st.session_state):
        feature_predict_select = st.session_state['target']
        st.write('Restored target ...')
    if('feature' in st.session_state):
        feature_input_select = st.session_state['feature']
        st.write('Restored feature input ...')
        
    if (X_train is None):
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_predict_select = st.selectbox(
            label='Select variable to predict',
            options=list(df.select_dtypes(include='number').columns),
            key=f"{df}",
            #index=8
        )

        st.session_state['target'] = feature_predict_select

        feature_input_select = st.multiselect(
            label='Select features for regression input',
            options=[f for f in list(df.select_dtypes(
                include='number').columns) if f != feature_predict_select],
            #key='feature_multiselect'
            key=f"{df}multi"
        )

        st.session_state['feature'] = feature_input_select

        st.write('You selected input {} and output {}'.format(
            feature_input_select, feature_predict_select))

        df = df.dropna()
        X = df.loc[:, df.columns.isin(feature_input_select)]
        Y = df.loc[:, df.columns.isin([feature_predict_select])]

        X = np.asarray(X.values.tolist()) 
        Y = np.asarray(Y.values.tolist()) 

        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1,key=f"{df}num")

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number)
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

def load_dataset(filepath):
    data = pd.read_csv(filepath)
    st.session_state['house_df'] = data
    return data

random.seed(10)
###################### FETCH DATASET #######################
df = None
if('house_df' in st.session_state):
    df = st.session_state['house_df']
else:
    filepath = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
    if(filepath):
        df = load_dataset(filepath)

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data(df)
    st.markdown("## Performance Evaluation")
    metric_options = ['mean_absolute_error',
                      'root_mean_squared_error', 'r2_score']    
    # Select multiple metrics for evaluation
    metric_select = st.multiselect(
        label='Select metrics for regression model evaluation',
        options=metric_options,
    )
    if (metric_select):
        st.session_state['metric_select'] = metric_select
        st.write('You selected the following metrics: {}'.format(metric_select))

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
    trained_models = [
        model for model in regression_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models
    # Select a trained regression model for evaluation
    model_select = st.multiselect(
        label='Select trained regression models for evaluation',
        options=trained_models
    )

    if st.button('Evaluate your selected regression models'):
            models = [st.session_state[model] 
                      for model in model_select]
            train_result_dict = {}
            val_result_dict = {}
            for idx, model in enumerate(models):
                train_result_dict[model_select[idx]] = compute_eval_metrics(
                        X_train, y_train, model, metric_select)
                val_result_dict[model_select[idx]] = compute_eval_metrics(
                        X_val, y_val, model, metric_select)

            st.markdown('### Predictions on the training dataset')
            st.dataframe(train_result_dict)
            st.markdown('### Predictions on the validation dataset')
            st.dataframe(val_result_dict)
            
    st.markdown("### Learning Curve")
    if st.button('Plot Learning Curve'):
        for model in model_select:
            trained_model = st.session_state[model]
            fig, df = plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metric_select, model)
            st.plotly_chart(fig)
 
    st.markdown("### Cross-Validation Evaluation")
    n_folds = st.number_input('Select Number of Cross-Validation Folds', min_value=2, max_value=20, value=5, step=1)
    if st.button('Perform Cross-validation'):
        if model_select:
            cv_results = {}
            for model_name in model_select:
                trained_model = st.session_state[model_name]
                cv_scores = perform_cross_validation(X_train, y_train, trained_model, n_folds, metric_select)
                cv_results[model_name] = cv_scores
            st.markdown('#### Cross-validation Results')
            for model_name, scores in cv_results.items():
                st.write(f"**{model_name}**")
                for metric, score in scores.items():
                    st.write(f"{metric}: {score:.4f}")
                    
    st.markdown("### Residual Plot Visualization")
    if st.button('Generate Residual Plot', key = 'res_plot'):
        for model_name in model_select:
            trained_model = st.session_state[model_name]
            y_pred_train = trained_model.predict(X_train)
            y_pred_val = trained_model.predict(X_val)
            fig_train = plot_residuals(y_train, y_pred_train, f'{model_name} (Training Data)')
            st.plotly_chart(fig_train, use_container_width=True)
            fig_val = plot_residuals(y_val, y_pred_val, f'{model_name} (Validation Data)')
            st.plotly_chart(fig_val, use_container_width=True)
               