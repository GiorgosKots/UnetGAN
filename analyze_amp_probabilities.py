def calculate_percentages(file_path, type="random forest"):
    # Initialize counters
    total = 0
    above_0_5 = 0
    above_0_8 = 0

    # Open and read the file
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            # Split each line into parts
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # Extract the probability and convert it to a float
                probability = float(parts[2])
                total += 1

                # Check if the probability is above 0.5 or 0.8
                if probability > 0.5:
                    above_0_5 += 1
                if probability > 0.8:
                    above_0_8 += 1

    # Calculate the percentages
    percentage_above_0_5 = (above_0_5 / total) * 100
    percentage_above_0_8 = (above_0_8 / total) * 100

    # Print the results
    print(f"Percentage above 0.5 ({type}): {percentage_above_0_5:.2f}%")
    print(f"Percentage above 0.8 ({type}): {percentage_above_0_8:.2f}%")

# Example usage:
# calculate_percentages("D:\GANs\GANs\campr4\CAMPdownload_2025-05-21 22-28-41.txt")