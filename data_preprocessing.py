from sklearn.model_selection import train_test_split


def preprocess_data(df, return_dates=False):
    df = df.dropna()
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    X = df[['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if return_dates:
        return X_train, X_test, y_train, y_test, X_train.index, X_test.index
    return X_train, X_test, y_train, y_test