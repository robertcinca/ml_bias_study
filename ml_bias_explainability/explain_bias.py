import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ml_bias_explainability.helpers import Helper


class ExplainBias:
    """An API class designed to explain model bias"""

    def __init__(
        self,
        training_type,
        features_of_interest,
        output,
        best_params,
        unique_values=10,
        feature_to_remove=None,
    ):
        self.training_type = training_type
        self.feature_to_remove = feature_to_remove
        self.features_of_interest = features_of_interest
        self.best_params = best_params
        self.batch_size = self.best_params["batch_size"]
        self.unique_values = unique_values
        self.output = output

    def sensitivity_analysis(
        self,
        model,
        raw_test_df,
        test_df,
    ):

        # Get columns for sensitivity analysis. All numeric + non-numeric cols below threshold
        object_list, numeric_list = Helper.get_df_types_list(raw_test_df)
        reduced_object_list = list(
            object_list[raw_test_df[object_list].nunique() <= self.unique_values]
        )

        column_list = list(numeric_list) + reduced_object_list

        if self.output in column_list:
            column_list.remove(self.output)

        # There is no point conducting sensitivity analysis on a feature that isn't in the model
        if self.feature_to_remove and self.feature_to_remove in column_list:
            column_list.remove(self.feature_to_remove)

        print("\nRunning the sensitivity analysis feature by feature. This might take a while.")
        returned_data = Parallel(n_jobs=-1, prefer="threads", verbose=8)(
            delayed(self._change_prediction_by_column)(column, model, raw_test_df, test_df)
            for column in column_list
        )

        list_of_predictions_dfs = [item[0] for item in returned_data]
        list_of_evaluation_dfs = [item[1] for item in returned_data]

        raw_predictions_df = pd.concat(list_of_predictions_dfs)
        predictions_change_df = self._measure_prediction_changes(raw_predictions_df)
        volatility_df = self._measure_volatility(raw_predictions_df)
        sensitivity_evaluation_df = pd.concat(list_of_evaluation_dfs)

        training_type_value = (
            f"{self.training_type}_{self.feature_to_remove}"
            if self.feature_to_remove
            else self.training_type
        )

        raw_predictions_df.insert(loc=0, column="training_type", value=training_type_value)
        predictions_change_df.insert(loc=0, column="training_type", value=training_type_value)
        volatility_df.insert(loc=0, column="training_type", value=training_type_value)

        return (
            raw_predictions_df,
            predictions_change_df,
            volatility_df,
            sensitivity_evaluation_df,
        )

    def _change_prediction_by_column(
        self,
        column,
        model,
        raw_test_df,
        test_df,
    ):
        column_predictions_df_list, evaluation_df_list = [], []

        if self.feature_to_remove:
            test_ds = Helper.df_to_dataset(
                Helper.remove_certain_features(test_df, self.feature_to_remove),
                batch_size=self.batch_size,
            )
        else:
            test_ds = Helper.df_to_dataset(test_df, batch_size=self.batch_size)

        predictions_original_array = model.predict(test_ds)
        labels_true = test_df[self.output]

        # if column is numeric, perturbe value, otherwise cycle through values
        is_numeric = False
        if raw_test_df[column].dtype.kind in "biufc":
            is_numeric = True
            max_col_val = test_df[column].max()
            min_col_val = test_df[column].min()
            step = max_col_val - min_col_val  # normalize over values range
            column_values = [-step * 0.10, -step * 0.05, step * 0.05, step * 0.10]  # +-5%, +-10%

            if raw_test_df[column].dtype.kind in "biu":  # keep 'count' features as int
                column_values = [round(x, 0) for x in column_values]
        else:
            column_values = test_df[column].unique()

        for column_value in column_values:
            modified_df = test_df.copy()
            if is_numeric:
                value_original = "x"  # np.repeat("x", len(raw_test_df))
                value_new = f"{column_value} from the original"
                modified_df[column] = modified_df[column].apply(
                    lambda x: round(max(min(x + column_value, max_col_val), min_col_val), 2)
                )
            else:
                value_original = raw_test_df.loc[test_df.index.tolist()][column].to_numpy()
                id_value = test_df[column].eq(column_value).idxmax()
                value_new = raw_test_df.loc[id_value][column]
                modified_df[column] = column_value

            if self.feature_to_remove:
                modified_test_ds = Helper.df_to_dataset(
                    Helper.remove_certain_features(modified_df, self.feature_to_remove),
                    batch_size=self.batch_size,
                )
            else:
                modified_test_ds = Helper.df_to_dataset(modified_df, batch_size=self.batch_size)

            predictions_array = model.predict(modified_test_ds)

            evaluation_rows = self._compute_evaluation_metrics(
                modified_df, model, raw_test_df, labels_true, predictions_array, column, value_new
            )

            sensitivity_evaluation_partial_df = pd.DataFrame(
                evaluation_rows,
                columns=[
                    "training_type",
                    "feature",
                    "feature_value",
                    "column_name",
                    "column_value",
                    "sample_count",
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                    "equalized_odds",
                    "equal_opportunity",
                    "statistical_parity",
                    "treatment_equality",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ],
            )

            labels_predicted_original = np.argmax(predictions_original_array, axis=1)
            labels_predicted_new = np.argmax(predictions_array, axis=1)
            prediction_difference_arrays = predictions_array - predictions_original_array
            standard_deviations = np.std(prediction_difference_arrays, axis=1)
            standard_deviation_directions = np.argmax(prediction_difference_arrays, axis=1)

            column_predictions_partial_df = pd.DataFrame(
                {
                    "column_name": column,
                    "value_original": value_original,
                    "value_new": value_new,
                    "label_true": labels_true,
                    "label_predicted_original": labels_predicted_original,
                    "label_predicted_new": labels_predicted_new,
                    "standard_deviation": standard_deviations,
                    "standard_deviation_direction": standard_deviation_directions,
                },
            )

            column_predictions_df_list.append(column_predictions_partial_df)
            evaluation_df_list.append(sensitivity_evaluation_partial_df)

        return [pd.concat(column_predictions_df_list), pd.concat(evaluation_df_list)]

    def _compute_evaluation_metrics(
        self, modified_df, model, raw_test_df, labels_true, predictions_array, column, value_new
    ):
        """
        Compute evaluation metrics
        """
        evaluation_rows = []

        for feature in self.features_of_interest:
            if feature == "all":
                feature_value_raw = "all"

                evaluation_json = Helper.evaluate_model(
                    labels_true,
                    predictions_array,
                    self.training_type,
                    feature,
                    feature_value_raw,
                    self.feature_to_remove,
                    modified_df,  # TEMP TO REMOVE
                )

                evaluation_rows.append(
                    {**evaluation_json, "column_name": column, "column_value": value_new}
                )
            else:
                feature_values = modified_df[feature].unique()

                for feature_value in feature_values:
                    sub_test_df = modified_df[modified_df[feature] == feature_value]

                    # remove columns before conversion to dataset
                    if self.feature_to_remove == feature:
                        sub_test_df = Helper.remove_certain_features(
                            sub_test_df, self.feature_to_remove
                        )

                    test_ds = Helper.df_to_dataset(sub_test_df, batch_size=self.batch_size)

                    id_value = modified_df[feature].eq(feature_value).idxmax()
                    feature_value_raw = raw_test_df.loc[id_value][feature]

                    evaluation_json = Helper.evaluate_model(
                        sub_test_df[self.output],
                        model.predict(test_ds),
                        self.training_type,
                        feature,
                        feature_value_raw,
                        self.feature_to_remove,
                        sub_test_df,  # TEMP TO REMOVE
                    )

                    evaluation_rows.append(
                        {
                            **evaluation_json,
                            "column_name": column,
                            "column_value": value_new,
                        }
                    )

        return evaluation_rows

    def _measure_prediction_changes(self, raw_predictions_df):
        """
        Measuring bias in sensitivity analysis: % samples where predicted output changes
        """
        aggregate_predictions_df = (
            raw_predictions_df.groupby(
                [
                    "column_name",
                    "value_original",
                    "value_new",
                    "label_predicted_original",
                    "label_predicted_new",
                ]
            )
            .size()
            .to_frame(name="size")
            .reset_index()
        )

        return aggregate_predictions_df

    def _measure_volatility(self, raw_predictions_df):
        """
        Measuring bias in sensitivity analysis: average volatility using mean of
        standard deviations of the difference between original and new prediction.
        """
        volatility_df = raw_predictions_df[
            raw_predictions_df["value_original"] != raw_predictions_df["value_new"]
        ]
        return (
            volatility_df.groupby(["column_name", "value_original", "value_new"])
            .agg(
                standard_deviation=("standard_deviation", "mean"),
                standard_deviation_direction=(
                    "standard_deviation_direction",
                    lambda x: x.value_counts().index[0],
                ),
            )
            .reset_index()
        )
