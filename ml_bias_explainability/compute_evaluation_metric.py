from ml_bias_explainability.helpers import Helper


class ComputeEvaluationMetric:
    def __init__(self, features_of_interest, output, batch_size):
        self.features_of_interest = features_of_interest
        self.output = output
        self.batch_size = batch_size

    def main(self, test_df, model, training_type, raw_test_df, feature_to_remove=None):
        """
        Computes evaluation metrics
        """
        evaluation_rows = []

        for feature in self.features_of_interest:
            if feature == "all":
                feature_value_raw = "all"

                test_ds = Helper.df_to_dataset(test_df, self.output, self.batch_size)

                evaluation_json = Helper.evaluate_model(
                    test_df[self.output],
                    model.predict(test_ds),
                    training_type,
                    feature,
                    feature_value_raw,
                    feature_to_remove,
                    test_df,  # TEMP TO REMOVE
                )

                evaluation_rows.append(evaluation_json)

            else:
                feature_values = test_df[feature].unique()

                for feature_value in feature_values:
                    sub_test_df = test_df[test_df[feature] == feature_value]

                    # remove columns before conversion to dataset
                    if feature_to_remove == feature:
                        sub_test_df = Helper.remove_certain_features(sub_test_df, feature_to_remove)

                    test_ds = Helper.df_to_dataset(sub_test_df, self.output, self.batch_size)

                    id_value = test_df[feature].eq(feature_value).idxmax()
                    feature_value_raw = raw_test_df.loc[id_value][feature]

                    evaluation_json = Helper.evaluate_model(
                        sub_test_df[self.output],
                        model.predict(test_ds),
                        training_type,
                        feature,
                        feature_value_raw,
                        feature_to_remove,
                        sub_test_df,  # TEMP TO REMOVE
                    )

                    evaluation_rows.append(evaluation_json)

        return evaluation_rows
