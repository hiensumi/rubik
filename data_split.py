import os

'''
This is a program that splits the dataset into training and validation sets.
'''

def split_data(csv_file, train_ratio=0.8):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(csv_file)
    train_data, val_data = train_test_split(data, train_size=train_ratio, random_state=42)

    train_data.to_csv('./image-dataset/rubik_coord_train.csv', index=False)
    val_data.to_csv('./image-dataset/rubik_coord_val.csv', index=False)
    
if __name__ == '__main__':
    split_data('./image-dataset/rubik_coord.csv', train_ratio=0.8)