from utils import data_processor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import os, sys, logging
import pandas as pd
import datetime

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# set up basic logging configuration, in real life logging configs should be stored and load from a yaml or ini
ROOT = logging.getLogger()
ROOT.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
ROOT.addHandler(handler)


def build_data_path(file):
    """
    Builds a full path to given file.
    :param file: string, name of data file
    :return: full path of file
    """
    return os.path.join(PROJECT_ROOT_DIR, "data", file)

def encoder(data):
    """
    Encode data to normalize types
    :param data: DataFrame
    :return: DataFrame
    """
    for col in data.columns:
        if data.dtypes[col] == "object":
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data

def predict(data):
    """
    Run binary logistic regression.
    :param data: DataFrame
    """

    data = encoder(data)

    # get feature values
    X = data.iloc[:, :-1]

    # get target values
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = LogisticRegression()

    # fit the model with data
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, predictions)
    print(cnf_matrix)

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


def load_and_process_raw_data(file):
    """
    Load and process training data.
    :param file: string
    :return: DataFrame
    """

    # get raw data file
    path = build_data_path(file)
    raw_data = data_processor.open_csv(path)
    schedule_status_column = pd.Series([])

    # fill in missing data
    raw_data["Project Phase Actual End Date"].fillna(0, inplace=True)
    raw_data["Project Phase Name"].fillna("None", inplace=True)

    # process input file and add new derived column, to add a boolean schedule status column
    for row_idx in range(len(raw_data)):
        estimated_end_date = raw_data["Project Phase Planned End Date"][row_idx]
        actual_end_date = raw_data["Project Phase Actual End Date"][row_idx]

        # DEBUG ME
        # print(row_idx)
        # print((estimated_end_date, actual_end_date))

        if (isinstance(estimated_end_date, str) and isinstance(actual_end_date, str)) and data_processor.is_date(estimated_end_date) and data_processor.is_date(actual_end_date):
            if estimated_end_date:
                estimated_end_date = datetime.datetime.strptime(estimated_end_date, '%m/%d/%Y')

            if actual_end_date:
                actual_end_date = datetime.datetime.strptime(actual_end_date, '%m/%d/%Y')

            if actual_end_date > estimated_end_date:
                schedule_status_column[row_idx] = True
            else:
                schedule_status_column[row_idx] = False
        else:
            schedule_status_column[row_idx] = True

    # insert new schedule status column data for all rows
    raw_data.insert(len(raw_data.head()), "Schedule Status", schedule_status_column)

    # check filled in values
    print(raw_data.count())

    return raw_data


if __name__ == "__main__":

    data_file = "ai_data_processed.csv"
    processed_data = load_and_process_raw_data(data_file)

    predict(processed_data)
