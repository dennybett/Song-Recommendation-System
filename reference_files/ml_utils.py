import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def train_test_split_marketing(df):
    X = df.drop(columns='y')
    y = df['y'].values.reshape(-1,1)
    return train_test_split(X, y, random_state=13)


# Functions for filling values
def fill_job(X):
    X['job'] = X['job'].fillna('unknown')
    return X

def fill_education(X):
    X['education'] = X['education'].fillna('primary')
    return X

def fill_contact(X):
    X['contact'] = X['contact'].fillna('unknown')
    return X

def fill_pdays(X):
    X['pdays'] = X['pdays'].fillna(-1)
    return X

def fill_poutcome(X):
    X['poutcome'] = X['poutcome'].fillna('nonexistent')
    return X

def fill_missing(X):
    X = fill_job(X)
    X = fill_education(X)
    X = fill_contact(X)
    X = fill_pdays(X)
    X = fill_poutcome(X)
    return X


# Functions for building and training encoders
def build_job_encoder(X_filled):
    job_encoder = OneHotEncoder(max_categories=5, handle_unknown='infrequent_if_exist', sparse_output=False)
    # Train the encoder
    job_encoder.fit(X_filled['job'].values.reshape(-1, 1))
    return {'column': 'job',
            'multi_col_output': True,
            'encoder': job_encoder}

def build_marital_encoder(X_filled):
    marital_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    marital_encoder.fit(X_filled['marital'].values.reshape(-1, 1))
    return {'column': 'marital',
            'multi_col_output': True,
            'encoder': marital_encoder}

def build_education_encoder(X_filled):
    education_encoder = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    education_encoder.fit(X_filled['education'].values.reshape(-1, 1))
    return {'column': 'education',
            'multi_col_output': False,
            'encoder': education_encoder}

def build_default_encoder(X_filled):
    default_encoder = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    default_encoder.fit(X_filled['default'].values.reshape(-1, 1))
    return {'column': 'default',
            'multi_col_output': False,
            'encoder': default_encoder}

def build_housing_encoder(X_filled):
    housing_encoder = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    housing_encoder.fit(X_filled['housing'].values.reshape(-1, 1))
    return {'column': 'housing',
            'multi_col_output': False,
            'encoder': housing_encoder}

def build_loan_encoder(X_filled):
    loan_encoder = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    loan_encoder.fit(X_filled['loan'].values.reshape(-1, 1))
    return {'column': 'loan',
            'multi_col_output': False,
            'encoder': loan_encoder}

def build_contact_encoder(X_filled):
    contact_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    contact_encoder.fit(X_filled['contact'].values.reshape(-1, 1))
    return {'column': 'contact',
            'multi_col_output': True,
            'encoder': contact_encoder}

def build_month_encoder(X_filled):
    month_encoder = OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']], handle_unknown='use_encoded_value', unknown_value=-1)

    # Train the encoder
    month_encoder.fit(X_filled['month'].values.reshape(-1, 1))
    return {'column': 'month',
            'multi_col_output': False,
            'encoder': month_encoder}

def build_poutcome_encoder(X_filled):
    poutcome_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Train the encoder
    poutcome_encoder.fit(X_filled['poutcome'].values.reshape(-1, 1))
    return {'column': 'poutcome',
            'multi_col_output': True,
            'encoder': poutcome_encoder}

def build_encoders(X_filled):
    encoder_functions = [build_job_encoder, 
                         build_marital_encoder, 
                         build_education_encoder, 
                         build_default_encoder,
                         build_housing_encoder,
                         build_loan_encoder,
                         build_contact_encoder,
                         build_month_encoder,
                         build_poutcome_encoder
                        ]
    return [encoder_function(X_filled) for encoder_function in encoder_functions]


# Encoding all categorical variables
def encode_categorical(X_filled, encoders):
    # Separate numeric columns
    dfs = [X_filled.select_dtypes(include='number').reset_index(drop=True)]

    single_col_encoders = []
    for encoder_dict in encoders:
        encoder = encoder_dict['encoder']
        column = encoder_dict['column']
        multi_col = encoder_dict['multi_col_output']
        if not multi_col:
            single_col_encoders.append(encoder_dict)
        else:
            dfs.append(pd.DataFrame(encoder.transform(X_filled[column].values.reshape(-1, 1)), columns=encoder.get_feature_names_out()))
    
    X_encoded = pd.concat(dfs, axis=1)

    for encoder_dict in single_col_encoders:
        encoder = encoder_dict['encoder']
        column = encoder_dict['column']
        multi_col = encoder_dict['multi_col_output']
        X_encoded[column] = encoder.transform(X_filled[column].values.reshape(-1, 1))

    return X_encoded

def build_target_encoder(y):
    encode_y = OneHotEncoder(drop='first', sparse_output=False)
    encode_y.fit(y)
    return encode_y

def encode_target(y, encode_y):
    
    return np.ravel(encode_y.transform(y))
