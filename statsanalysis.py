def calculate_PI(weights, counts):
    N = sum(counts)  # Total number of participants
    if N == 0:
        return 0  # Avoid division by zero
    
    PI = sum(w * n for w, n in zip(weights, counts)) / N
    return PI

# Example usage
weights = [1, 2, 3, 4, 5]  # Example weights for discomfort levels
counts = [10, 12, 0, 0, 0]  # Number of participants selecting each level

PI_value = calculate_PI(weights, counts)
print(f"Perception Index (PI): {PI_value:.3f}")
