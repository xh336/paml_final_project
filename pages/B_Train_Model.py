import numpy as np                
import pandas as pd               
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st             
import random
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

#############################################

st.title('Train Model')

#############################################

# Checkpoint 1
def split_dataset(X, y, number,random_state=45):
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(number/100), random_state=random_state)
    
    return X_train, X_val, y_train, y_val

class LinearRegression(object) : 
    def __init__(self, learning_rate=0.001, num_iterations=500): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.cost_history=[]

    # Checkpoint 2: Hypothetical function h(x) 
    def predict(self, X): 
        '''
        Make a prediction using coefficients self.W and input features X
        Y=X*W
        
        Input: X is matrix of column-wise features
        Output: prediction of house price
        '''
        self.W=(self.W).reshape(-1,1)
        self.num_examples, _ = X.shape
        X_transform = np.append(np.ones((self.num_examples, 1)), X, axis=1)
        prediction = np.dot(X_transform, self.W)
        return prediction

    # Checkpoint 3: Update weights in gradient descent 
    def update_weights(self):     
        '''
        Update weights of regression model by computing the 
        derivative of the RSS cost function with respect to weights
        
        Input: None
        Output: None
        ''' 
        # m: no_of_training_examples, n:no_of_features 
        self.num_examples, _ = (self.X).shape
        X_transform = np.append(np.ones((self.num_examples, 1)), self.X, axis=1)
        
        # Make prediction using fitted line
        Y_pred = LinearRegression.predict(self, self.X) 
        
        # calculate gradients using RMSE: RMSE = sqrt((1/n)sum_{num_examples} error^2) 
        # derivative wrt w
        dW = - np.dot((2 * (X_transform.T)), (self.Y - Y_pred)) / self.num_examples

        cost = mean_squared_error(self.Y, Y_pred, squared=False)
        
        # update weights 
        self.W = self.W - self.learning_rate * dW 

        # store cost
        self.cost_history.append(cost)

        return self
    
    # Checkpoint 4: Model training 
    def fit(self, X, Y): 
        self.num_examples, self.num_features = X.shape

        # weight, featues X, and output Y initialization 
        self.W = np.zeros(self.num_features + 1) # +1 for const offset 
        X = LinearRegression.normalize(self, X)
        self.X = X
        self.Y = Y

        # Run Gradient Descent
        for _ in range(self.num_iterations): 
            LinearRegression.update_weights(self) 
        return self
    
    # Helper function
    def normalize(self, X):
        '''
        Standardize features X by column

        Input: X is input features (column-wise)
        Output: Standardized features by column
        '''
        X_normalized=X
        try:
            means = np.mean(X, axis=0) #columnwise mean and std
            stds = np.std(X, axis=0)+1e-7
            X_normalized = (X-means)/(stds)
        except ValueError as err:
            st.write({str(err)})
        return X_normalized
    
    # Checkpoint 5: Return regression coefficients
    def get_weights(self, model_name, features):
        '''
        This function prints the coefficients of the trained models
        
        Input:
            - 
        Output:
            - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - 'Multiple Linear Regression'
            - 'Polynomial Regression'
            - 'Ridge Regression'
            - 'Lasso Regression'
        '''
        out_dict = {'Multiple Linear Regression': []}
        for i in range(len(features)):
            out_dict[model_name] = self.W
        return out_dict

# Helper functions
def load_dataset(filepath):
    '''
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    '''
    try:
        data = pd.read_csv(filepath)
        st.session_state['house_df'] = data
    except ValueError as err:
            st.write({str(err)})
    return data

random.seed(10)
###################### FETCH DATASET #######################
# Use file_uploader to upload the dataset locally
df=None

filename = 'datasets/study_performance.csv'
if('house_df' in st.session_state):
    df = st.session_state['house_df']
else:
    if(filename):
        df = load_dataset(filename)

###################### DRIVER CODE #######################

if df is not None:
    # Display dataframe as table
    st.dataframe(df.describe())

    # Select variable to predict
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=list(df.select_dtypes(include='number').columns),
        key='feature_selectbox'
        ## index=8
    )

    st.session_state['target'] = feature_predict_select

    # Select input features
    all_features = [f for f in list(df.columns) if f != feature_predict_select]
    feature_input_select = st.multiselect(
        label='Select features for regression input',
        options=all_features,
        key='feature_multiselect'
    )
    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(feature_input_select, feature_predict_select))
    df = df.dropna()
    X = df.loc[:,df.columns.isin(feature_input_select)]
    Y = df.loc[:,df.columns.isin([feature_predict_select])]

    # Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    split_number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    # Compute the percentage of test and training data
    X_train_df, X_val_df, y_train_df, y_val_df = split_dataset(X, Y, split_number)
    st.session_state['X_train_df'] = X_train_df
    st.session_state['X_val_df'] = X_val_df
    st.session_state['y_train_df'] = y_train_df
    st.session_state['y_val_df'] = y_val_df

    # Convert to numpy arrays
    X = np.asarray(X.values.tolist()) 
    Y = np.asarray(Y.values.tolist()) 
    X_train, X_val, y_train, y_val = split_dataset(X, Y, split_number)
    train_percentage = (len(X_train) / (len(X_train)+len(y_val)))*100
    test_percentage = (len(X_val)) / (len(X_train)+len(y_val))*100

    st.markdown('Training dataset ({1:.2f}%): {0:.2f}'.format(len(X_train),train_percentage))
    st.markdown('Test dataset ({1:.2f}%): {0:.2f}'.format(len(X_val),test_percentage))
    st.markdown('Total number of observations: {0:.2f}'.format(len(X_train)+len(y_val)))
    train_percentage = (len(X_train)+len(y_train) /
                        (len(X_train)+len(X_val)+len(y_train)+len(y_val)))*100
    test_percentage = ((len(X_val)+len(y_val)) /
                        (len(X_train)+len(X_val)+len(y_train)+len(y_val)))*100

    #regression_methods_options = ['Multiple Linear Regression']
    # Collect ML Models of interests
    #regression_model_select = st.multiselect(
        #label='Select regression model for prediction',
        #options=regression_methods_options,
    #)
    regression_model_select = 'Multiple Linear Regression'
    st.write('The model used for this application: {}'.format(
        regression_model_select))

    # Multiple Linear Regression
    if (regression_model_select == 'Multiple Linear Regression'):
        st.markdown('#### ' + 'Multiple Linear Regression Training')

        # Add parameter options to each regression method
        learning_rate_input = st.text_input(
            label='Input learning rate 👇',
            value='0.4',
            key='mr_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(learning_rate_input))

        num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='200',
            key='mr_iter_textinput'
        )
        st.write('You select the following number of iteration value(s): {}'.format(num_iterations_input))

        multiple_reg_params = {
            'num_iterations': [float(val) for val in num_iterations_input.split(',')],
            'alpha': [float(val) for val in learning_rate_input.split(',')]
        }

        if st.button('Train Multiple Linear Regression Model'):
            # Handle errors
            try:
                multi_reg_model = LinearRegression(learning_rate=multiple_reg_params['alpha'][0], 
                                                   num_iterations=int(multiple_reg_params['num_iterations'][0]))
                multi_reg_model.fit(X_train, y_train)
                st.session_state['Multiple Linear Regression'] = multi_reg_model
            except ValueError as err:
                st.write({str(err)})

        if 'Multiple Linear Regression' not in st.session_state:
            st.write('Multiple Linear Regression Model is untrained')
        else:
            st.write('Multiple Linear Regression Model trained')

    # Plot model
    st.markdown('##### ' + 'Plot of Real vs Prediction')
    plot_model = 'Multiple Linear Regression'
    # Select input features
    feature_plot_select = st.selectbox(
        label='Select feature to plot',
        options=feature_input_select
    )
    
    if(regression_model_select and plot_model and feature_plot_select):
        if(plot_model in st.session_state):
            find_feature = np.char.find(feature_input_select, feature_plot_select)
            f_idx = np.where(find_feature == 0)[0][0]
            feature_name = feature_input_select[f_idx]
            
            model = st.session_state[plot_model]

            y_pred = model.predict(X_val)
            if(y_pred is not None):

                test = X_val[:,f_idx]
                test = test.reshape(-1)
                y_pred = y_pred.reshape(-1)
                y_val = y_val.reshape(-1)

                fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
                
                fig.add_trace(go.Scatter(x=test,
                            y=y_val, mode='markers', name="Features"), row=1, col=1)
                fig.add_trace(go.Line(x=test,
                            y=y_pred, mode='markers', name="Predictions"), row=1, col=1)

                fig.update_xaxes(title_text="X")
                fig.update_yaxes(title_text='Y', row=0, col=1)
                fig.update_layout(title='Projection of predictions with real values '+feature_plot_select)
                st.plotly_chart(fig)
    
    # Store models
    trained_models={}
    trained_models['Multiple Linear Regression'] = st.session_state['Multiple Linear Regression']

    # Inspect Regression coefficients
    st.markdown('##### Inspect model coefficients')
    models = {}
    weights_dict = {}
    models['Multiple Linear Regression'] = st.session_state['Multiple Linear Regression']
    weights_dict = models['Multiple Linear Regression'].get_weights('Multiple Linear Regression', feature_input_select)
    st.write(weights_dict)

    # Select multiple models to inspect
    st.markdown('##### Inspect model cost')
    inspect_model_cost = 'Multiple Linear Regression'

    if(inspect_model_cost):
        try:
            fig = make_subplots(rows=1, cols=1,
                shared_xaxes=True, vertical_spacing=0.1)
            cost_history=trained_models[inspect_model_cost].cost_history

            x_range = st.slider("Select x range:",
                                    value=(0, len(cost_history)))
            st.write("You selected : %d - %d"%(x_range[0],x_range[1]))
            cost_history_tmp = cost_history[x_range[0]:x_range[1]]
            
            fig.add_trace(go.Scatter(x=np.arange(x_range[0],x_range[1],1),
                        y=cost_history_tmp, mode='markers', name=inspect_model_cost), row=1, col=1)

            fig.update_xaxes(title_text="Training Iterations")
            fig.update_yaxes(title_text='Cost', row=1, col=1)
            fig.update_layout(title=inspect_model_cost)
            st.plotly_chart(fig)
        except Exception as e:
            print(e)

    st.write('Continue to Evaluation Methods')