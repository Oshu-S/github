import matplotlib.pyplot as plt
import numpy as np

# Define the x values
x = np.linspace(-10, 10, 100)

# Define the linear equation y = mx + c (e.g., y = 2x + 1)
m = 2  # Slope
c = 1  # Intercept
y = m * x + c

# Plot the line
plt.plot(x, y, label=f"y = {m}x + {c}", color='blue')

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Linear Plot")
plt.legend()

# Show the plot
plt.grid()
plt.show()
print("hey")
