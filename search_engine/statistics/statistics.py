import numpy as np

def linear_regression_residuals(df, X_columns, Y_column, adjusted=False):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Ensure that X_columns exist in the dataframe
    if not all(item in df.columns for item in X_columns):
        raise ValueError('Not all specified X_columns are in the dataframe.')
    if Y_column not in df.columns:
        raise ValueError('The Y_column is not in the dataframe.')

    # Prepare the feature matrix X by selecting the X_columns and adding an intercept term
    X = df[X_columns].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term

    # Extract the target variable vector Y
    Y = df[Y_column].values

    # Calculate theta using the pseudo-inverse
    # theta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    # Make predictions
    # Y_pred = X @ theta
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    # Calculate residuals
    residuals = Y - Y_pred
    # Add residuals to the dataframe
    df['residuals'] = residuals

    # Calculate R-squared
    SS_res = (residuals ** 2).sum()
    SS_tot = ((Y - np.mean(Y)) ** 2).sum()
    R_squared = 1 - SS_res / SS_tot
    
    if adjusted:
        # Calculate Adjusted R-squared
        n = X.shape[0]  # Number of observations
        p = X.shape[1] - 1  # Number of predictors, excluding intercept
        R_squared = 1 - ((1 - R_squared) * (n - 1)) / (n - p - 1)

    return df, R_squared
