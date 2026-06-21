import numpy as np
import matplotlib.pyplot as plt

# data
data = np.array([
    [85.0, 8.0, 1.50, 300, 7.0],   # Cotton-1
    [87.5, 8.5, 1.52, 340, 6.5],   # Cotton-2
    [88.0, 8.3, 1.51, 320, 6.8],   # Cotton-3
    [86.5, 7.8, 1.50, 310, 7.2],   # Cotton-4
    [70.0, 8.0, 1.48, 550, 2.0],   # Hemp-1
    [74.0, 8.0, 1.49, 690, 1.6],   # Hemp-2
    [77.0, 7.5, 1.48, 820, 1.8],   # Hemp-3
    [73.0, 7.9, 1.47, 600, 2.1],   # Hemp-4
    [71.0, 10.0, 1.50, 810, 1.8],  # Flax-1
    [75.0, 10.5, 1.52, 900, 1.5],  # Flax-2
    [78.0, 11.0, 1.53, 1050, 1.4], # Flax-3
    [73.0, 10.2, 1.51, 850, 1.7],  # Flax-4
    [46.0, 9.5,  1.44, 140, 3.5],  # Canola-1
    [48.0, 9.8,  1.44, 160, 3.8],  # Canola-2
    [50.0, 10.2, 1.45, 180, 4.0],  # Canola-3
    [47.0, 9.6,  1.44, 150, 3.6],  # Canola-4
    [65.0, 12.0, 1.44, 400, 1.8],  # Jute-1
    [67.0, 12.5, 1.45, 440, 2.0],  # Jute-2
    [65.5, 12.0, 1.44, 410, 1.9],  # Jute-3
    [66.0, 12.2, 1.45, 430, 2.0],  # Jute-4
])

x = data[:,0] # cellulose content
y = data[:,3] # tensile strength
y_bar = np.mean(y)
m = len(x)


# model 1: linear
A1 = np.column_stack([np.ones(m),x])
ATA1 = A1.T @ A1
ATy1 = A1.T @ y
c1 = np.linalg.solve(ATA1, ATy1)

print("A^T A=", ATA1)
print("\nA^T y=", ATy1)
print("\nLinear model coefficients [c1,c2]:", c1)


# model 1 scatterplot
y_pred_linear = A1 @ c1
plt.scatter(x,y,label="Data", color="gray")
plt.plot(x,y_pred_linear, color="deeppink", label="Linear Fit")
plt.xlabel("Cellulose Content (%)")
plt.ylabel("Tensile Strength (MPa)")
plt.title("Linear Least Squares Fit: Cellulose Content vs Tensile Strength")
plt.legend()
plt.show()


# model 2: degree-4 polynomial
A2 = np.column_stack([x**0, x**1, x**2, x**3, x**4])
ATA2 = A2.T @ A2
ATy2 = A2.T @ y
c2 = np.linalg.solve(ATA2, ATy2)

print("First 4 rows of A:")
print(A2[:4])
print("\nA^T y=", ATy2)
print("\nPolynomial model coefficients [c1...c5]:", c2)


# model 2 scatterplot
x_plot = np.linspace(np.min(x), np.max(x), 300)
A2_plot = np.column_stack([np.ones_like(x_plot),
                           x_plot, x_plot**2, x_plot**3,
                           x_plot**4])
y_plot = A2_plot @ c2
plt.scatter(x,y,label="Data", color="gray")
plt.plot(x_plot, y_plot, color="deeppink", label="Polynomial Fit (Degree 4)")
plt.xlabel("Cellulose Content (%)")
plt.ylabel("Tensile Strength (MPa)")
plt.title("Polynomial Least Squares Fit: Cellulose Content vs Tensile Strength")
plt.legend()
plt.show()


# R^2 values
y_bar = np.mean(y)
SS_tot = (y - y_bar) @ (y - y_bar)

# linear R^2
y_hat1 = A1 @ c1
SS_res1 = (y - y_hat1) @ (y - y_hat1)
R2_1 = 1 - SS_res1 / SS_tot
print("R^2 (linear model):", R2_1)

# polynomial R^2
y_hat2 = A2 @ c2
SS_res2 = (y - y_hat2) @ (y - y_hat2)
R2_2 = 1 - SS_res2 / SS_tot
print("\nR^2 (polynomial model):", R2_2)


# PCA
X_centered = data - np.mean(x, axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
print("Singular values:\n", S)
print("\nPrincipal directions (V):", Vt.T)

# projection onto first 2 PCs
Z = X_centered @ Vt.T[:, :2]
print("\nZ:", Z[:5])