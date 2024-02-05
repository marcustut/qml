import polars as pl
from sklearn.model_selection import train_test_split

DATASET_PATH = "datasets/iris.csv"

def load_iris(path: str):
    return pl.read_csv(path)

if __name__ == "__main__":
    data = load_iris(DATASET_PATH)
    print(data)
