import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw03_data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start = minimum_value, stop = maximum_value, step = 0.001)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    num_bins = len(left_borders)
    bin_values = np.zeros(num_bins)

    for bin_idx in range(num_bins):
        indices_in_bin = (left_borders[bin_idx] < x_train) & (x_train <= right_borders[bin_idx])
        
        if np.any(indices_in_bin):
            bin_values[bin_idx] = np.mean(y_train[indices_in_bin])
        else:
            bin_values[bin_idx] = 0

    num_queries = len(x_query)
    y_hat = np.zeros(num_queries)

    for query_idx in range(num_queries):
        query_point = x_query[query_idx]
        for bin_idx in range(num_bins):
            if left_borders[bin_idx] < query_point <= right_borders[bin_idx]:
                y_hat[query_idx] = bin_values[bin_idx]
                break
    # your implementation ends above
    return(y_hat)
    
bin_width = 0.35
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    y_hat = np.zeros(x_query.shape)
    for i, x in enumerate(x_query):
        distances = np.abs(x - x_train)
        weights = (distances < bin_width / 2).astype(float)
        
        numerator = np.sum(weights * y_train)
        denominator = np.sum(weights)
        
        y_hat[i] = numerator / np.maximum(denominator, 1e-10)
    # your implementation ends above
    return(y_hat)

bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    y_hat = np.zeros(x_query.shape)

    for idx, query_point in enumerate(x_query):
        scaled_distances = (query_point - x_train) / bin_width
        gaussian_weights = np.exp(-0.5 * scaled_distances**2) / math.sqrt(2 * math.pi)

        weighted_sum = np.sum(gaussian_weights * y_train)
        total_weight = np.sum(gaussian_weights)

        y_hat[idx] = weighted_sum / total_weight
        # your implementation ends above
    return(y_hat)


bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
