import itertools
import json
import re
import statistics as s

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Helper:
    @staticmethod
    def read_json(file_path):
        with open(file_path) as stream:
            return json.load(stream)

    @staticmethod
    def read_yaml(file_path):
        with open(file_path) as stream:
            return yaml.safe_load(stream)

    @staticmethod
    def convert_types(df):
        """
        Convert df to tensor-accepted types. Numeric followed by categorical columns
        """
        # Replace bool with numeric
        df.replace({False: 0.0, True: 1.0}, inplace=True)

        object_list, numeric_list = Helper.get_df_types_list(df)

        # convert numeric types
        df[numeric_list] = df[numeric_list].astype(np.float32)

        # convert objects
        for column in object_list:
            df[column] = pd.Categorical(df[column])
            df[column] = df[column].cat.codes
        return df

    @staticmethod
    def get_df_types_list(df):
        """
        Gets a list of columns that are numeric and objects
        """
        numeric_list = df.select_dtypes(include=[np.number]).columns
        object_list = df.select_dtypes(object).columns
        return object_list, numeric_list

    @staticmethod
    def get_input_column_dtypes(dataframe):
        numeric_list = list(dataframe.select_dtypes(include=[np.number]).columns)
        object_list = list(dataframe.select_dtypes(object).columns)

        if "target" in numeric_list:
            numeric_list.remove("target")
        elif "target" in object_list:
            object_list.remove("target")

        print(f"List of numeric features for this dataset: {numeric_list}")
        print(f"List of non-numeric features: {object_list}\n")

        return numeric_list, object_list

    @staticmethod
    def define_output(dataframe, output):
        # Define target
        dataframe["target"] = dataframe[output]

        # Drop un-used columns.
        dataframe = dataframe.drop(columns=[output])

        # name new output
        output = "target"

        return dataframe, output

    @staticmethod
    def train_test_split(dataframe):
        train, test = train_test_split(dataframe, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        print(f"Total number of training samples: {len(train)}")
        print(f"Total number of validation samples: {len(val)}")
        print(f"Total number of test samples: {len(test)}\n")

        return train, test, val

    @staticmethod
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        """
        A utility method to create a tf.data dataset from a Pandas Dataframe
        """
        df = dataframe.copy()

        output_raw = np.array(df.pop("target"))
        label_encoder = LabelEncoder()
        output_encoded = label_encoder.fit_transform(output_raw)
        labels = to_categorical(output_encoded)  # defining as 2D output tensor

        df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}

        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    @staticmethod
    def remove_certain_features(df, feature_to_remove):
        return df.drop(columns=[feature_to_remove], errors="ignore")

    @staticmethod
    def create_neuron_layer_combinations(neurons, hidden_layers):
        # TODO: see if to remove this function
        layer_sizes = list(itertools.product(neurons, repeat=1))
        for i in range(2, hidden_layers + 1, 1):
            layer_sizes += list(itertools.product(neurons, repeat=i))

        return layer_sizes

    @staticmethod
    def evaluate_model(
        labels_true,
        predictions_array,
        training_type,
        feature,
        feature_value_raw,
        feature_to_remove=None,
        test_df=None,  # TEMP TO REMOVE
    ):
        """
        Measuring bias in sensitivity analysis: computing summary statistics
        """
        y_pred_bool = np.argmax(predictions_array, axis=1)

        evaluation_dict = classification_report(labels_true, y_pred_bool, output_dict=True)

        (
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        ) = Helper.raw_performance_values(labels_true.values, y_pred_bool)

        equalized_odds = Helper.equalized_odds(
            true_positives, false_positives, true_negatives, false_negatives
        )

        equal_opportunity = Helper.equal_opportunity(
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        )

        statistical_parity = Helper.statistical_parity(
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        )

        treatment_equality = Helper.treatment_equality(
            false_positives,
            false_negatives,
        )

        return {
            "training_type": (
                f"{training_type}_{feature_to_remove}" if feature_to_remove else training_type
            ),
            "feature": feature,
            "feature_value": feature_value_raw,
            "sample_count": evaluation_dict["weighted avg"]["support"],
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "equalized_odds": equalized_odds,  # definition 1
            "equal_opportunity": equal_opportunity,  # definition 2
            "statistical_parity": statistical_parity,  # definition 3
            "treatment_equality": treatment_equality,  # definition 6
            "accuracy": evaluation_dict["accuracy"],
            "precision": evaluation_dict["weighted avg"]["precision"],
            "recall": evaluation_dict["weighted avg"]["recall"],
            "f1": evaluation_dict["weighted avg"]["f1-score"],
        }

    @staticmethod
    def raw_performance_values(y_actual, y_hat):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        sample_count = len(y_hat)

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                true_positives += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                false_positives += 1
            if y_actual[i] == y_hat[i] == 0:
                true_negatives += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                false_negatives += 1

        return (
            true_positives / sample_count,
            false_positives / sample_count,
            true_negatives / sample_count,
            false_negatives / sample_count,
        )

    @staticmethod
    def equalized_odds(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Equalized Odds
        """
        return s.harmonic_mean(
            [
                (true_positives)
                / (true_positives + false_positives + true_negatives + false_negatives),
                (false_positives)
                / (true_positives + false_positives + true_negatives + false_negatives),
            ]
        )

    @staticmethod
    def equal_opportunity(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Equal Opportunity
        """
        return (true_positives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )

    @staticmethod
    def statistical_parity(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Statistical Parity
        """
        return (true_positives + false_positives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )

    @staticmethod
    def treatment_equality(
        false_positives,
        false_negatives,
    ):
        """
        Fairness Definition: Treatment Equality
        """
        return false_positives / (1 + false_negatives)

    @staticmethod
    def strip_text_for_file_naming(text):
        # remove any special characters
        text = re.sub(r"\W", " ", text)

        # Substituting multiple spaces with single space
        text = re.sub(r"\s+", " ", text, flags=re.I)

        # Converting to Lowercase
        text = text.lower()

        # Remove leading and ending spaces
        text = text.strip()

        # Replace space with _ to meet formatting requirements of file naming
        text = text.replace(" ", "_")

        return text
