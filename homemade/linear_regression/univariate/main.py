import numpy as np
import pandas as pd
from homemade.utils.features.prepare_for_training import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, normalize_data=True):
        (
            data_processed,
            features_mean,
            features_deviation
        ) = prepare_for_training(data, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iteration=500):
        cost_history = self.gradient_descent(alpha, num_iteration)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iteration):
        cost_history = []
        for _ in range(num_iteration):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        num_examples = self.data.shape[0]
        predictions = LinearRegression.hypothesis(self.data, self.theta)
        delta = predictions - self.labels
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (delta.T @ self.data).T
        self.theta = theta

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(
            data,
            self.normalize_data,
        )[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        data_processed = prepare_for_training(
            data,
            self.normalize_data,
        )[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions

    def cost_function(self, data, labels):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(data, self.theta) - labels

        cost = (1 / 2 * num_examples) * (delta.T @ delta)

        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        predictions = data @ theta
        return predictions


if __name__ == '__main__':
    data = pd.read_csv(
        '/Users/anhvietpham/Documents/Dev-Chicken/Machine-Learning/machine-learning-research/homemade/linear_regression/data/2019.csv')
    input_param_name = "GDP per capita"
    out_param_name = "Score"
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)
    x_train = train_data[[input_param_name]].values
    y_train = train_data[[out_param_name]].values
    x_test = test_data[[input_param_name]].values
    y_test = test_data[[out_param_name]].values
    num_iterations = 500
    regularization_param = 0
    learning_rate = 0.01
    linear_regression = LinearRegression(x_train, y_train)
    # Train linear regression.
    (theta, cost_history) = linear_regression.train(learning_rate, num_iterations)
    # Print training results.
    print('Initial cost: {:.2f}'.format(cost_history[0]))
    print('Optimized cost: {:.2f}'.format(cost_history[-1]))
    train_cost = linear_regression.get_cost(x_train, y_train)
    test_cost = linear_regression.get_cost(x_test, y_test)
    print(f'Train cost: {train_cost}')
    print(f'Test cost: {test_cost}')
    test_predictions = linear_regression.predict(x_test)
    test_predictions_table = pd.DataFrame({
        'Economy GDP per Capita': x_test.flatten(),
        'Test Happiness Score': y_test.flatten(),
        'Predicted Happiness Score': test_predictions.flatten(),
        'Prediction Diff': (y_test - test_predictions).flatten()
    })
    print(test_predictions_table.head(10))
