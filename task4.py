import math
import pandas as pd
import numpy as np

def calculate_log_likelihood(fico_scores, defaults, bucket_boundaries):
    """
    Note: fico_scores and defaults are lists of the same length, where fico_scores[i] corresponds to the FICO score of
    the i-th borrower and defaults[i] is 1 if the i-th borrower defaulted and 0 otherwise.
    """
    num_buckets = len(bucket_boundaries) - 1
    num_records_per_bucket = np.zeros(num_buckets, dtype=int)
    num_defaults_per_bucket = np.zeros(num_buckets, dtype=int)
    
    # Assign each FICO score to a bucket
    for i in range(len(fico_scores)):
        fico = fico_scores[i]
        for j in range(num_buckets):
            if bucket_boundaries[j] <= fico < bucket_boundaries[j + 1]:
                num_records_per_bucket[j] += 1
                if defaults[i] == 1:
                    num_defaults_per_bucket[j] += 1
                break

    log_likelihood = 0
    
    # Iterate through each bucket
    for i in range(num_buckets):
        bucket_size = bucket_boundaries[i + 1] - bucket_boundaries[i]
        if bucket_size == 0:
            continue
        
        # Calculate the probability of default in the current bucket
        if num_records_per_bucket[i] != 0:
            probability_default = num_defaults_per_bucket[i] / num_records_per_bucket[i]
        else:
            probability_default = 0
        
        # Calculate the log likelihood contribution of the current bucket
        if probability_default != 0 and probability_default != 1:
            log_likelihood += num_defaults_per_bucket[i] * math.log(probability_default) + \
                               (num_records_per_bucket[i] - num_defaults_per_bucket[i]) * math.log(1 - probability_default)
    
    return log_likelihood

def dynamic_programming_quantization(fico_scores, defaults, num_buckets_range):
    """
    Perform dynamic programming to find the optimal bucket boundaries that maximize the log likelihood of the data.
    """
    # Base case: when there is only one bucket within a given range of FICO scores
    if num_buckets_range == 1:
        # Calculate the log likelihood for a single bucket encompassing the entire range of FICO scores
        min_fico = min(fico_scores)
        max_fico = max(fico_scores)
        bucket_boundaries = [min_fico, max_fico]
        log_likelihood = calculate_log_likelihood(fico_scores, defaults, bucket_boundaries)
        
        # Return the bucket boundaries and the corresponding log likelihood
        return bucket_boundaries, log_likelihood
    # Sort the FICO scores in ascending order
    sorted_fico_scores = sorted(fico_scores)

    # Recursive step
    # Initialize variables to store the optimal solution
    best_log_likelihood = float('-inf')
    best_bucket_boundaries = None
    
    # Iterate through possible positions to split the range of FICO scores
    for i in range(1, len(sorted_fico_scores)):
        print(f"Number of buckets: {num_buckets_range}, Splitting at index: {i}")
        # Divide the FICO scores into two subranges
        fico_scores_left = sorted_fico_scores[:i]
        fico_scores_right = sorted_fico_scores[i:]
        
        # Recursively find the optimal bucket boundaries for each subrange
        _, log_likelihood_left = dynamic_programming_quantization(fico_scores_left, defaults, num_buckets_range - 1)
        _, log_likelihood_right = dynamic_programming_quantization(fico_scores_right, defaults, num_buckets_range - 1)
        
        # Calculate the combined log likelihood for the current split
        combined_log_likelihood = log_likelihood_left + log_likelihood_right
        
        # Update the best solution if the combined log likelihood is greater
        if combined_log_likelihood > best_log_likelihood:
            best_log_likelihood = combined_log_likelihood
            best_bucket_boundaries = sorted_fico_scores[i - 1:i + 1]
    
    # Return the optimal bucket boundaries and the corresponding log likelihood
    return best_bucket_boundaries, best_log_likelihood



if __name__ == "__main__":
    # read csv data for task 4
    data = pd.read_csv("Task_3_and_4_Loan_Data.csv")
    print("Data read successfully!")
    # only keep FICO score and default columns
    fico_scores = data["fico_score"].values
    defaults = data["default"].values
    # Perform dynamic programming quantization
    num_buckets_range = 3
    bucket_boundaries, log_likelihood = dynamic_programming_quantization(fico_scores, defaults, num_buckets_range)
    print("Optimal bucket boundaries:", bucket_boundaries)
    print("Log likelihood:", log_likelihood)
    