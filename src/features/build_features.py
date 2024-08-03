from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=1234):
    x = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
