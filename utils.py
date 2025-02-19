#General dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import sklearn

# Models to use in unsupervised learning pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


# Models to use in supervised learning pipeline
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso   
from sklearn.linear_model import Ridge   
from sklearn.svm import SVR

#Preprocessing dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

#metrics
from sklearn.metrics import mean_squared_error, r2_score

random_state = 42

def eda(df):
    """
    Initial investigation of the data
    Args:  
    df : A pandas dataframe object containing data of interest
     
    """
    data_info = df.info()
    null_cols = df.isna().sum()
    summary_stats = df.describe()
    
    return data_info, null_cols

def process_data(df, features, target):
    
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    features : A list of columns containing all the independent variables(predictors)
    target : A list of column(s) of the dependent variable(label)
    
    Returns: An array of X and y cleaned and encoded.
    """
    
    df = df.dropna().reset_index(drop=True)
    #features
    X = df.drop(target, axis=1)
    #target
    y = df[target]

    #Identify categorical columns and encode them
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Apply one-hot encoding to categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, dtype=float)

    return X_encoded, y
    
def plot_numeric_distributions(df):
    """
    Plot numerical features in the dataframe to identify patterns or problems with the data
    Args:
    DataFrame : A pandas dataframe object containing data to be trained
    
    Returns : histogram distribution, box plot and probplot of numerical columns in the dataframe
    """
    
    df = df.fillna(0)
    #select only numerical columns from the dataframe
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    #plot only cols that have at least 10 unique numerical values
    plot_cols = [col for col in numeric_cols if df[col].nunique() > 10]
    n_rows = len(plot_cols)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4*n_rows))

    print("DISTRIBUTION OF DATA IN NUMERICAL COLUMNS")
    for i, col in enumerate(plot_cols):
            # Histogram distribution
            sns.histplot(df[col], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Distribution of {col}', fontsize=15)   
        
            # Box plot to detect outliers
            sns.boxplot(x=df[col], ax=axes[i, 1])
            axes[i, 1].set_title(f'Box Plot of {col}', fontsize=15)    
        
            # Q-Q plot comparing our observed quantiles with the theorectical ideal distribution
            stats.probplot(df[col], dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'Q-Q Plot of {col}', fontsize=15)
    
    plt.tight_layout()
    plt.show()
    


def plot_correlation_heatmap(df):
    
    """
    Plot correlation heatmap of numerical features in the Dataframe
    
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    
    Returns : correlation heatmap of numerical columns in the dataframe
    """

    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of numerical columns')
    plt.show()
    
    
# Create a function to calculate Variance Inflation Factor
def calc_vif(X):
    
    """ Calculates the variance inflation factor for each column (features)
    Args:
    features: A list of columns containing all the independent variables
    
    Returns : A Dataframe of VIF calculated in descending order
    
    """
    
    #select only numerical columns to avoid inf errors
    numeric_cols = X.select_dtypes(include=['int', 'float']).columns
    X = X[numeric_cols].dropna()
    
    #Create df to store each feature and their calculated vif
    vif = pd.DataFrame()
    vif["features"] = numeric_cols

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(0, X.shape[1])]
    
    return vif.sort_values(by="VIF", ascending=False)
   


def r2_adj(X, y, model):
    """
    Calculates adjusted r-squared values 
    
    Args:
    X: features/Independent variables, the data to fit
    y: dependent variable, the target data to try to predict
    model: The estimator or object to use to train the data
    
    Returns: adjusted r squared value accounting for number of predictors
    """
    r2 = model.score(X,y)
    n = X.shape[0]
    p = y.ndim
    
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)  

#Model with imputed data (replaced null values with the mean calculated in each column)
def model_generator_imputed(df, features, target):
    
    X = df[features]
    y = df[target]
    
    # Identify columns with null values
    null_columns = X.columns[X.isnull().any()]
    
    #impute each null value with the average computed in that column
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    for col in null_columns:
        X[col] = imp.fit_transform(X[[col]])
    
    
    #One hot encode the categorical variables in this imputed DataFrame
    X_imp = pd.get_dummies(X)
    X_imp_train, X_imp_test, y_train, y_test = train_test_split(X_imp, y, random_state=random_state)
    
    models = {
        "LR": LinearRegression(),
        "SGD":SGDRegressor(),
        "SVR": SVR()
    }
    
    #Dictionary to store all the different performance metrics called
    results = {}
    
    #Loop over every acronym and model in the dictionary
    for name, model in models.items():
        #Transform the data by scaling it first before running the model
        pipeline = Pipeline([
            ("Scale", StandardScaler(with_mean=False)),
            (name, model)
        ])
        #Train the model
        pipeline.fit(X_imp_train, y_train)
        #Make predictions using the model
        y_pred = pipeline.predict(X_imp_test)
        
        #Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2_value = r2_score(y_test, y_pred)
        r2_adj_value = r2_adj(X_imp_test, y_test, pipeline)

        results["R squared"] = r2_value
        results["Adjusted R squared"] = r2_adj_value
        results["Mean Squared error"] = mse
        print(f"{model} || R squared: {r2_value:.3f} || Adjusted R squared:{r2_adj_value:.3f}")
        print("===================================================================")
    
#Model on data after dropping null values
def model_generator(df, features, target):
    
    X_encoded, y = process_data(df, features, target)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=random_state)
    
    models = {
        "LR": LinearRegression(),
        "SGD":SGDRegressor(),
        "SVR": SVR()
    }
    
    #Dictionary to store all the different performance metrics called
    results = {} 
    #Loop over every acronym and model in the dictionary
    for name, model in models.items():
        
        #Transform the data by scaling it first before running the model
        pipeline = Pipeline([
            ("Scale", StandardScaler(with_mean=False)),
            (name, model)
        ])
        #Train the model
        pipeline.fit(X_train, y_train)
        #Make predictions with the model
        y_pred = pipeline.predict(X_test)
        
        #Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2_value = r2_score(y_test, y_pred)
        r2_adj_value = r2_adj(X_test, y_test, pipeline)

        results["R squared"] = r2_value
        results["Adjusted R squared"] = r2_adj_value
        results["Mean Squared error"] = mse
        print(f"{model}|| R squared: {r2_value:.3f} || Adjusted R squared:{r2_adj_value:.3f}")
        print("===================================================================")

#
# Clustering models to use in the unsupervised learning pipeline
#

# Train KMeans model 
def train_kmeans_model(df, num_clusters):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    num_clusters : The number of clusters to form
    
    Returns: DataFrame with A KMeans model trained on the data.
    """
    
    # Initialize a KMeans model
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=random_state)
    
    # Fit the model to the data
    kmeans.fit(df)

    # Make predictions on the data
    clusters = kmeans.predict(df)
    
    # Add the cluster labels to the original DataFrame
    df['Cluster'] = clusters

    return df

# Train Hierarchical Clustering model 
def train_hierarchical_clustering_model(df, num_clusters):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    num_clusters : The number of clusters to form
    
    Returns: A Hierarchical Clustering model trained on the data.
    """
    
    # Initialize a Hierarchical Clustering model
    hc = AgglomerativeClustering(n_clusters=num_clusters)
    
    # Fit the model to the data
    hc.fit(df)

    # Make predictions on the data
    clusters = hc.fit_predict(df)
    
    # Add the cluster labels to the original DataFrame
    df['Cluster'] = clusters

    return df
    
# Train Gaussian Misture Model 
def train_gaussian_mixture_model(df, num_components):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    num_components : The number of components to use in the Gaussian Mixture Model
    
    Returns: A Gaussian Mixture Model trained on the data.
    """
    
    # Initialize a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_components, random_state=random_state)
    
    # Fit the model to the data
    gmm.fit(df)

    # Make predictions on the data
    clusters = gmm.predict(df)
    
    # Add the cluster labels to the original DataFrame
    df['Cluster'] = clusters

    return df

# Function to plot the clusters 
def plot_2_clusters(df, features):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    features : A list of columns containing all the independent variables(predictors)
    
    Returns: A scatter plot showing the clusters formed by the KMeans model.
    """
    
    # Plot the clusters
    plt.figure(figsize=(10, 10))
    if (features == "kmeans"):
        plt.scatter(df['pca1'], df['pca2'], c=df[f"kmeans_labels"], cmap='viridis')
        plt.title('KMeans Clustering')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig('output/kmeans_pca.png')
        plt.show()
    elif (features == "agglo"):
        plt.scatter(df['pca1'], df['pca2'], c=df[f"agglo_labels"], cmap='viridis')
        plt.title('Agglomerative Clustering')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig('output/agglo_pca.png')
        plt.show()
    elif (features == "gaussian"):
        plt.scatter(df['pca1'], df['pca2'], c=df[f"gaussian_labels"], cmap='viridis')
        plt.title('Gaussian Mixture Model Clustering')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig('output/gaussian_pca.png')
        plt.show()

# Elbow method to determine the optimal number of clusters 
def elbow_method(df, max_clusters):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    max_clusters : The maximum number of clusters to form
    
    Returns: A plot showing the elbow curve for the KMeans model.
    """
    
    # Initialize a list to store the sum of squared distances for each k
    ssd = []

    # Loop over a range of k values
    for k in range(1, max_clusters + 1):
        # Create a KMeans model with k clusters
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        
        # Fit the model to the data
        kmeans.fit(df)
        
        # Append the sum of squared distances to the list
        ssd.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), ssd, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    return ssd

# Determine the rate of decrease between each k value 
def rate_of_decrease(ssd):
    """
    Args:
    ssd : A list of sum of squared distances for each k value
    
    Returns: A list of rate of decrease between each k value.
    """
    
    # Calculate the rate of decrease between each k value
    rate_of_decrease = [ssd[i] - ssd[i + 1] for i in range(len(ssd) - 1)]

    for i in range(len(ssd) - 1):
        print(f"Rate of decrease between k={i + 1} and k={i + 2}: {rate_of_decrease[i]}")
    
    return rate_of_decrease

