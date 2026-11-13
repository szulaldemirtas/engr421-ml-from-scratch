import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.5
maximum_value = 5.1
step_size = 0.001
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    # your implementation starts below
    feature_count = X_train.shape[1]  # Total number of features (D)
    total_data_points = len(y_train)  # Total number of training points (N)

    node_indices[1] = np.arange(total_data_points)  # Initialize the root node with all data indices
    is_terminal[1] = False  # Root node is not terminal initially
    need_split[1] = True  # Root node needs to be split

    while any(need_split.values()):
        # Collect nodes that require splitting
        nodes_to_split = [node_id for node_id, to_split in need_split.items() if to_split]

        for current_node in nodes_to_split:
            # Retrieve data indices corresponding to the current node
            data_ids = node_indices[current_node]
            need_split[current_node] = False  # Mark this node as processed

            # Calculate mean for the current node
            node_means[current_node] = np.mean(y_train[data_ids])

            # Stop splitting if pre-pruning condition is met
            if len(data_ids) <= P:
                is_terminal[current_node] = True
                continue

            # Initialize variables for finding the best split
            split_errors = np.zeros(feature_count)
            optimal_splits = np.zeros(feature_count)

            for feature_idx in range(feature_count):  # Iterate over all features
                # Unique values in the current feature for this node
                unique_vals = np.sort(np.unique(X_train[data_ids, feature_idx]))
                
                # Midpoints between unique values as candidate split points
                candidate_splits = (unique_vals[1:] + unique_vals[:-1]) / 2
                split_sse = np.zeros(len(candidate_splits))

                for split_idx, split_val in enumerate(candidate_splits):
                    # Split data indices into left and right subsets
                    left_ids = data_ids[X_train[data_ids, feature_idx] > split_val]
                    right_ids = data_ids[X_train[data_ids, feature_idx] <= split_val]

                    # Skip invalid splits
                    if len(left_ids) == 0 or len(right_ids) == 0:
                        continue

                    # Compute SSE for left and right subsets
                    left_mean = np.mean(y_train[left_ids])
                    right_mean = np.mean(y_train[right_ids])

                    split_sse[split_idx] = (
                        np.sum((y_train[left_ids] - left_mean) ** 2) + 
                        np.sum((y_train[right_ids] - right_mean) ** 2)
                    )

                # Store the best split for the current feature
                split_errors[feature_idx] = np.min(split_sse)
                optimal_splits[feature_idx] = candidate_splits[np.argmin(split_sse)]

            # Select the feature with the lowest error for splitting
            best_feature = np.argmin(split_errors)
            best_split_value = optimal_splits[best_feature]

            # Save the split information
            node_features[current_node] = best_feature
            node_splits[current_node] = best_split_value

            # Create child nodes and split data
            left_ids = data_ids[X_train[data_ids, best_feature] > best_split_value]
            right_ids = data_ids[X_train[data_ids, best_feature] <= best_split_value]

            # Update child nodes in dictionaries
            node_indices[2 * current_node] = left_ids
            node_indices[2 * current_node + 1] = right_ids

            is_terminal[current_node] = False
            need_split[2 * current_node] = True
            need_split[2 * current_node + 1] = True
    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)

# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    num_points = len(X_query)  # Total number of query data points
    y_hat = np.zeros(num_points)  # Array to store predicted values for each query point

    for point_idx in range(num_points):
        current_node = 1  # Begin traversal at the root node

        while True:
            # Check if the current node is terminal
            if is_terminal[current_node]:
                y_hat[point_idx] = node_means[current_node]  # Assign mean value for the terminal node
                break  # Exit loop as the prediction for this point is complete

            # Determine the direction to move (left or right child)
            feature_index = node_features[current_node]
            split_threshold = node_splits[current_node]

            if X_query[point_idx, feature_index] > split_threshold:
                current_node = 2 * current_node  # Move to the left child
            else:
                current_node = 2 * current_node + 1  # Move to the right child
    # your implementation ends above
    return(y_hat)

# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    # Helper function to build the path of rules from a terminal node to the root
    def build_rule_path(node, node_features, node_splits):
        path_rules = []  # List to collect conditions
        current = node  # Start from the terminal node

        while current > 1:  # Traverse up until the root node
            parent = current // 2  # Parent node
            feature = node_features[parent] + 1  # Convert feature index to 1-based
            split_value = node_splits[parent]

            # Add condition based on whether it's a left or right child
            if current % 2 == 0:  # Left child: greater than condition
                path_rules.append(f"x{feature} > {split_value:.2f}")
            else:  # Right child: less than or equal condition
                path_rules.append(f"x{feature} <= {split_value:.2f}")

            current = parent  # Move to the parent node

        return list(reversed(path_rules))  # Reverse rules to go from root to leaf


    # Identify all terminal nodes and print rules
    leaf_nodes = [node for node, terminal in is_terminal.items() if terminal]

    for leaf_node in leaf_nodes:
        rules = build_rule_path(leaf_node, node_features, node_splits)  # Get path of rules
        rules_string = " AND ".join(rules)  # Combine rules into a single string

        # Print final rules for each terminal node
        print(f"Node {leaf_node:02}: {rules_string} => {node_means[leaf_node]:.2f}")
    # your implementation ends above

P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)
