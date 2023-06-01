import numpy as np

features = np.array([(1, 2, 3), (2, 3, 4)])
targets = [3, 4]
features_test = np.array([(3, 4, 5)])
targets_test = [5]

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape

last_loss = None

# 输入层权重
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden))

# 隐藏层权重
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features, targets):

      ## Forward pass ##
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))

      ## Backward pass ##
        # loss 误差
        error = y - output

        # 输出层求导，
        output_error_term = error * output * (1 - output)
        hidden_error = np.dot(output_error_term, weights_hidden_output)
        # Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output

        # 隐藏层求导
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        del_w_input_hidden += hidden_error_term * x[:, None] # x.T

    # 更新权重
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
