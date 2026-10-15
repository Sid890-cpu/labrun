import random

# Toy "experiment": estimate pi by Monte Carlo
n = 100000
inside = sum(1 for _ in range(n) if random.random() ** 2 + random.random() ** 2 <= 1)
print("pi_estimate", 4 * inside / n)
