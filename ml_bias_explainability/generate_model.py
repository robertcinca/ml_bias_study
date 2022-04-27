import numpy as np
import tensorflow as tf
from keras.constraints import maxnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from ml_bias_explainability.helpers import Helper


class GenerateModel:
    """
    Needs the output feature, the best parameters for the model and the encoded datasets.

    Returns a compiled model.
    """

    def __init__(self, best_params):
        self.best_params = best_params

    def main(self, encoded_features, all_inputs, train_ds, val_ds, test_ds):
        all_features = tf.keras.layers.concatenate(encoded_features)

        # Initial layer
        x = tf.keras.layers.Dense(
            self.best_params["neurons"],
            kernel_initializer=self.best_params["init_mode"],
            kernel_constraint=maxnorm(self.best_params["weight_constraint"]),
            activation=self.best_params["activation"],
        )(all_features)

        # Inner layers
        for i in range(self.best_params["hidden_layers"] - 1):
            x = tf.keras.layers.Dense(
                self.best_params["neurons"],
                kernel_initializer=self.best_params["init_mode"],
                kernel_constraint=maxnorm(self.best_params["weight_constraint"]),
                activation=self.best_params["activation"],
            )(x)

        # Dropout rate
        x = tf.keras.layers.Dropout(self.best_params["dropout_rate"])(x)

        # Output
        matrix_size, activation, loss = self._get_output_size(train_ds)
        output = tf.keras.layers.Dense(matrix_size, activation)(x)

        # Model processing
        model = tf.keras.Model(all_inputs, output)
        model.compile(
            optimizer=self.best_params["optimizer"],
            loss=loss,
            metrics=["accuracy"],
        )
        print("Training the model:")
        model.fit(train_ds, epochs=self.best_params["epochs"], validation_data=val_ds)

        loss, accuracy = model.evaluate(test_ds)
        print("Model accuracy is:", accuracy)

        return model

    def random_forest(self, train_df, test_df):
        # Split into X and Y
        train_df = Helper.convert_types(train_df.copy())
        test_df = Helper.convert_types(test_df.copy())

        y_train = train_df["target"]
        X_train = train_df.drop(columns=["target"])

        y_test = test_df["target"]
        X_test = test_df.drop(columns=["target"])

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=50)]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 60, num=11)]
        max_depth.append(None)  # type: ignore
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        _bootstrap = [True, False]
        # Create the random grid
        random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": _bootstrap,
        }

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 5 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_model = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=50,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        # Fit the random search model
        rf_model.fit(X_train, y_train)

        print("Best random forest parameters:")
        print(rf_model.best_params_)

        classifier = rf_model.best_estimator_
        y_pred = classifier.predict(X_test)

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Accuracy metrics:")
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

        return rf_model

    def _get_output_size(self, ds):
        [(train_features, label_batch)] = ds.take(1)

        matrix_size = len(label_batch[0])

        if matrix_size == 1:
            activation = "sigmoid"
            loss = "binary_crossentropy"
        else:
            activation = "softmax"
            loss = "categorical_crossentropy"

        return matrix_size, activation, loss
