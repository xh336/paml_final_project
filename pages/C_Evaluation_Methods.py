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
        out_dict = {'Polynomial Regression': [],
                'Ridge Regression': [],
                   'Lasso Regression': []}
        for i in range(len(features)):
            out_dict[model_name] = self.W
        return out_dict
class PolynomailRegression(LinearRegression):
    def __init__(self, degree, learning_rate, num_iterations):
        self.degree = degree

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    # Helper function
    def transform(self, X):
        '''
        Converts a matrix of features for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n

        Input:
            - 
        Output:
            -
        '''
        try:
            # coverting 1D to 2D
            if X.ndim==1:
                X = X[:,np.newaxis]
            num_examples, num_features = X.shape
            features = [np.ones((num_examples, 1))] # for bias, the first column
            # initialize X_transform
            for j in range(1, self.degree + 1):
                # For better understanding see doc: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
                for combinations in itertools.combinations_with_replacement(range(num_features), j): # this will give us the combination of features
                    feature = np.ones(num_examples)
                    for each_combination in combinations:
                        feature = feature * X[:,each_combination]
                    features.append(feature[:, np.newaxis]) # collecting list of arrays each array is the feature
            # concating the list of feature in each column them
            X_transform = np.concatenate(features, axis=1)
        except ValueError as err:
            st.write({str(err)})
        return X_transform
    
    # Checkpoint 6: Model training
    def fit(self, X, Y):
        '''
        Use gradient descent to update the weights for self.num_iterations

        Input:
            - X: Input features X
            - Y: True values of housing prices
        Output: None
        '''

        self.num_examples, self.num_features = X.shape
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)
        # X_normalize = LinearRegression.normalize(self, X_transform)
        self.W = np.ones(X_transform.shape[1]).reshape(-1,1)
        
        for _ in range(self.num_iterations):
            Y_pred = self.predict(X)
            dW = - (2 * (X_normalize.T).dot(Y - Y_pred) ) / self.num_examples
            self.W = self.W - self.learning_rate * dW
            cost= np.sqrt(np.sum(np.power(Y-Y_pred,2))/len(Y_pred)) 
            self.cost_history.append(cost)
        return self
    
    # Checkpoint 7: Make a prediction with Polynomial Regression model
    def predict(self, X):
        '''
        Make a prediction using coefficients self.W and input features X
        Y=X*W
        
        Input: X is matrix of column-wise features
        Output: prediction of house price
        '''
        X_transform = self.transform(X)
        X_normalize = LinearRegression.normalize(self, X_transform)
        prediction = X_normalize.dot(self.W)
        return prediction

# Ridge Regression 
class RidgeRegression(LinearRegression): 
    def __init__(self, learning_rate, num_iterations, l2_penalty): 
        self.l2_penalty = l2_penalty 

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    # Checkpoint 8: Update weights in gradient descent 
    def update_weights(self):      
        '''
        Update weights of regression model by computing the 
        derivative of the RSS + l2_penalty*w cost function with respect to weights

        Input: None
        Output: None
        '''
        self.num_examples, self.num_features = (self.X).shape
        X_transform = np.append(np.ones((self.num_examples, 1)), self.X, axis=1)
        Y_pred = self.predict(self.X) 
        
        dW = - ( 2 * ( X_transform.T ).dot( self.Y - Y_pred )  +  2 * self.l2_penalty * self.W ) / self.num_examples

        cost = mean_squared_error(self.Y, Y_pred, squared=False)
        
        # update weights 
        self.W = self.W - self.learning_rate * dW 

        # store cost
        self.cost_history.append(cost)
        return self
class LassoRegression(LinearRegression):
    def __init__(self, learning_rate, num_iterations, lambda_param):
        self.lambda_param = lambda_param
        LinearRegression.__init__(self, learning_rate, num_iterations)
    def update_weights(self):
        '''
        Update weights using gradient descent with L1 regularization (Lasso)

        The gradient of the L1 term is the sign of the weight, which needs to be handled separately
        from the usual gradient of the least squares term.
        '''
        X_transform = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        predictions = self.predict(self.X)
        # Gradient of the loss (without regularization)
        dW = -2 * np.dot(X_transform.T, (self.Y - predictions)) / self.X.shape[0]
        
        # Apply L1 regularization, except for the bias term
        dW[1:] += self.lambda_param * np.sign(self.W[1:])
        
        # Update weights
        self.W -= self.learning_rate * dW
        # Compute and store the cost with regularization term
        cost = mean_squared_error(self.Y, predictions, squared=False) + self.lambda_param * np.linalg.norm(self.W[1:], 1)
        self.cost_history.append(cost)
        return self
  

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

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    df = df.dropna()
    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin([feature_predict_select])]

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

    regression_methods_options = ['Polynomial Regression', 
                                  'Ridge Regression',
                                 'Lasso Regression']
    # Collect ML Models of interests
    regression_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=regression_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        regression_model_select))
    # Polynomial Regression
    if (regression_methods_options[0] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[0])

        poly_degree = st.number_input(
            label='Enter the degree of polynomial',
            min_value=0,
            max_value=1000,
            value=3,
            step=1,
            key='poly_degree_numberinput'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_degree))

        poly_num_iterations_input = st.number_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
            key='poly_num_iter'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_num_iterations_input))

        poly_input=[0.001]
        poly_learning_rate_input = st.text_input(
            label='Input learning rate 👇',
            value='0.0001',
            key='poly_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(poly_learning_rate_input))

        poly_reg_params = {
            'num_iterations': poly_num_iterations_input,
            'alphas': [float(val) for val in poly_learning_rate_input.split(',')],
            'degree' : poly_degree
        }

        if st.button('Train Polynomial Regression Model'):
            # Handle errors
            try:
                poly_reg_model = PolynomailRegression(poly_reg_params['degree'], 
                                                      poly_reg_params['alphas'][0], 
                                                      poly_reg_params['num_iterations'])
                poly_reg_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[0]] = poly_reg_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[0] not in st.session_state:
            st.write('Polynomial Regression Model is untrained')
        else:
            st.write('Polynomial Regression Model trained')

    # Ridge Regression
    if (regression_methods_options[1] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[1])

        # Add parameter options to each regression method
        ridge_l2_penalty_input = st.text_input(
            label='Enter the l2 penalty (0-1)👇',
            value='0.5',
            key='ridge_l2_penalty_textinput'
        )
        st.write('You select the following l2 penalty value(s): {}'.format(ridge_l2_penalty_input))

        ridge_num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='ridge_num_iter'
        )
        st.write('You set the number of iterations to: {}'.format(ridge_num_iterations_input))

        ridge_alphas = st.text_input(
            label='Input learning rate 👇',
            value='0.0001',
            key='ridge_lr_textinput'
        )
        st.write('You select the following learning rate: {}'.format(ridge_alphas))

        ridge_params = {
            'num_iterations': [int(val) for val in ridge_num_iterations_input.split(',')],
            'learning_rate': [float(val) for val in ridge_alphas.split(',')],
            'l2_penalty':[float(val) for val in ridge_l2_penalty_input.split(',')]
        }
        if st.button('Train Ridge Regression Model'):
            # Train ridge on all feature --> feature selection
            # Handle Errors
            try:
                ridge_model = RidgeRegression(learning_rate=ridge_params['learning_rate'][0],
                                           num_iterations=ridge_params['num_iterations'][0],
                                           l2_penalty=ridge_params['l2_penalty'][0])
                ridge_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[1]] = ridge_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[1] not in st.session_state:
            st.write('Ridge Model is untrained')
        else:
            st.write('Ridge Model trained')

    if (regression_methods_options[2] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[2])

        # Add parameter options to each regression method
        lambda_input = st.text_input(
            label='Enter the lambda input (0-1)👇',
            value='0.1',
            key='lambda_textinput'
        )
        st.write('You select the following Lambda Value(s): {}'.format(lambda_input))

        lasso_num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='500',
            key='lasso_num_iter'
        )
        st.write('You set the number of iterations to: {}'.format(lasso_num_iterations_input))

        lasso_alphas = st.text_input(
            label='Input learning rate 👇',
            value='0.001',
            key='lasso_lr_textinput'
        )
        st.write('You select the following learning rate: {}'.format(lasso_alphas))

        lasso_params = {
            'num_iterations': [int(val) for val in lasso_num_iterations_input.split(',')],
            'learning_rate': [float(val) for val in lasso_alphas.split(',')],
            'lambda':[float(val) for val in lambda_input.split(',')]
        }
        if st.button('Train Lasso Regression Model'):
            # Train ridge on all feature --> feature selection
            # Handle Errors
            try:
                lasso_model = LassoRegression(learning_rate=lasso_params['learning_rate'][0],
                                           num_iterations=lasso_params['num_iterations'][0],
                                           lambda_param=lasso_params['lambda'][0])
                lasso_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[2]] = lasso_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[2] not in st.session_state:
            st.write('Lasso Model is untrained')
        else:
            st.write('Lasso Model trained')

    
    st.markdown('#### Inspect fitted model')
    # Plot model
    plot_model = st.selectbox(
        label='Select model to plot',
        options=regression_model_select,
        key='plot_model_select'
    )

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
    for model_name in regression_model_select:
        if(model_name in st.session_state):
            trained_models[model_name] = st.session_state[model_name]

    # Inspect Regression coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select model',
        options=regression_model_select,
        key='inspect_multiselect'
    )
    st.write('You selected the {} models'.format(inspect_models))
    
    models = {}
    weights_dict = {}
    if(inspect_models):
        st.write('You are in')
        for model_name in inspect_models:
            if(model_name in trained_models):
                models[model_name] = st.session_state[model_name]
                weights_dict = models[model_name].get_weights(model_name, feature_input_select)
        st.write(weights_dict)

    # Inspect model cost
    st.markdown('## Inspect model cost')

    # Select multiple models to inspect
    inspect_model_cost = st.selectbox(
        label='Select model',
        options=regression_model_select,
        key='inspect_cost_multiselect'
    )

    st.write('You selected the {} model'.format(inspect_model_cost))

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

    st.write('Continue to Test Model')