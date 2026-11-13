import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities1 = np.genfromtxt("hw06_predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("hw06_predicted_probabilities2.csv", delimiter = ",")

# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated thresholds of size (N + 1,)
def calculate_threholds(predicted_probabilities):
    # your implementation starts below

    boundaries = np.array([0, 1])
    extended_values = np.sort(np.concatenate((boundaries, predicted_probabilities)))

    thresholds = (extended_values[:-1] + extended_values[1:]) / 2
  
    # your implementation ends above
    return thresholds

thresholds1 = calculate_threholds(predicted_probabilities1)
print(thresholds1)

thresholds2 = calculate_threholds(predicted_probabilities2)
print(thresholds2)

# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the thresholds of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    # your implementation starts below
    fp_rates = np.zeros_like(thresholds)
    tp_rates = np.zeros_like(thresholds)

    total_positives = np.sum(true_labels == 1)
    total_negatives = np.sum(true_labels == -1)

    for idx, threshold in enumerate(thresholds):
        predicted_labels = np.where(predicted_probabilities > threshold, 1, -1)
       
        true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
        false_positives = np.sum((true_labels == -1) & (predicted_labels == 1))

        tp_rates[idx] = true_positives / total_positives if total_positives > 0 else 0.0
        fp_rates[idx] = false_positives / total_negatives if total_negatives > 0 else 0.0

    # your implementation ends above
    return fp_rates, tp_rates

fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, thresholds1)
print(fp_rates1[495:505])
print(tp_rates1[495:505])

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, thresholds2)
print(fp_rates2[495:505])
print(tp_rates2[495:505])

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()
plt.show()
fig.savefig("hw06_roc_curves.pdf", bbox_inches = "tight")

# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
    # your implementation starts below
    auroc = 0.0

    # Use the trapezoidal rule to calculate the area under the curve
    for idx in range(len(tp_rates) - 1):
        width = fp_rates[idx] - fp_rates[idx + 1]  # Difference in FP rates
        height = (tp_rates[idx] + tp_rates[idx + 1]) / 2  # Average of TP rates
        auroc += width * height  # Add the area of each trapezoid

    # your implementation ends above
    return auroc

auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))