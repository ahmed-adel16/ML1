import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === loading and splitting the data ===

def load_classification_data(filepath):
    df = pd.read_csv(filepath, header=None)  
    return df

def balance_classes(df):
    gammas = df[df.iloc[:, -1] == 'g'] # getting the data with 'g' class
    hadrons = df[df.iloc[:, -1] == 'h'] # getting the data with 'h' class

    gammas_sampled = gammas.sample(len(hadrons), random_state=42) # creating a sample of gammas with length of hadrons
    balanced_df = pd.concat([gammas_sampled, hadrons]) # concatenating both of hadrons and gammas back in one dataset
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffling 

def split_classification_data(df):
    X = df.iloc[:, :-1] # all data except the target class
    y = df.iloc[:, -1] # target class

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) # splitting the data into 70% train and 30% for validation and test (temp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # splitting the temp data into (15% test, 15% validation (from original dataset))

    scaler_X = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)



    return X_train, X_val, X_test, y_train, y_val, y_test # return all data splitted (train, validation, test)


def split_regression_data(df):
    X = df.drop(columns=['Median_House_Value'])  # Features
    y = data['Median_House_Value'].to_frame()  # Target, reshaped for scaler and convert it to dataframe to become 2d using (to_frame())

    # Train-Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale X (features)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    # Scale y (target variable)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)  # Fit on train only
    y_val = scaler_y.transform(y_val)  # Transform val
    y_test = scaler_y.transform(y_test)  # Transform test

    return X_train, X_val, X_test, y_train, y_val, y_test
