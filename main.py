from pandas.io.data import DataReader
from datetime import datetime
import os
import pandas as pd

def get_data_if_not_exists():
    if os.path.exists("./data/ibm.csv"):
        return pd.read_csv("./data/ibm.csv")
    else:
        if not os.path.exists("./data"):
            os.mkdir("data")
        ibm_data = DataReader('IBM',  'yahoo', datetime(1950,1,1), datetime.today())
        pd.DataFrame(ibm_data).to_csv("./data/ibm.csv")
        return pd.DataFrame(ibm_data)


if __name__ == "__main__":
    print get_data_if_not_exists()