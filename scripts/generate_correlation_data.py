import numpy as np
import os 
import csv 
import string 

# Create a new random number generator (RNG) instance
rng = np.random.default_rng()

# Generate random integers from low (inclusive) to high (exclusive)
# e.g., 0 to 9
def generate_sequences(num_sequences, length=32):
    rng = np.random.default_rng()
    characters = string.ascii_letters + string.digits + string.punctuation
    char_array = np.array(list(characters))
    
    # Generate random indices (1000 rows, 32 cols)
    random_indices = rng.integers(low=0, high=len(char_array), size=(num_sequences, length))
    
    # Map to characters and join
    char_matrix = char_array[random_indices]
    sequences = ["".join(row) for row in char_matrix]
    
    return sequences

def save_sequences_to_csv(sequences, filepath):
    """
    Writes sequences to a CSV at the specified path, creating directories if needed.
    """
    # 1. Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # 2. Write the CSV
    with open(filepath, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence"])  # Header
        
        for i, seq in enumerate(sequences):
            writer.writerow([i, seq])
    
    print(f"Success! {len(sequences)} sequences saved to '{filepath}'.")  

if __name__ == "__main__":
    output_path = "correlation_data/correlation_data.csv"
    
    my_data = generate_sequences(num_sequences=2000, length=32)
    
    # Save
    save_sequences_to_csv(my_data, output_path)

