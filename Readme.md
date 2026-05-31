# Practice 4: Function Approximation using a Radial Basis Function Neural Network (RBFNN)
---
## Project Overview
This project implements a Radial Basis Function Neural Network (RBFNN) from scratch using only low-level mathematical libraries (`NumPy). The primary objective is to approximate the continuous 2D mathematical function:

$$f(x,y) = \sin(\sqrt{x^2 + y^2})$$

within the domain $x \in [-5, 5]$ and $y \in [-5, 5]$.


---

## Network Architecture & Learning Phases

The RBF network architecture is divided into three distinct steps:
1. **Unsupervised Phase ($K$-Means):** Finds the coordinates of $k$ cluster centers (**centroids**) across the input data distribution. These centroids act as the focal points for the spatial activation.
2. **Hidden Layer Transformation (Gaussian RBF):** Transforms the 2D spatial coordinates into a non-linear $k$-dimensional feature space ($Z$) using Gaussian functions:
   $$\phi(x, c_i) = \exp\left(-\frac{\|x - c_i\|^2}{2\sigma_i^2}\right)$$
   where $\sigma_i$ represents the variance (spread) calculated dynamically based on the deviation of points within each cluster.
3. **Supervised Phase (ADALINE):** Trains a linear neuron using the **Widrow-Hoff Delta Rule**.

---
## Experimental Test Results

The network was evaluated across a wide spectrum of hidden layer sizes ($k$) to analyze model capacity against the test dataset's Mean Squared Error (MSE).

| Centroids ($k$) | Test MSE |
| :--- | :--- |
| **5** | 0.24901 |
| **10** | 0.15348 |
| **20** | 0.07362 |
| **40** | **0.05442** *(Classical Optimal)* |
| **80** | 0.07701 |
| **100** | 0.07712 |
| **120** | 0.09776 |
| **150** | 0.12052 |
| **180** | 0.11049 |
| **210** | 0.16420 *(Maximum Overfitting)* |
| **250** | 0.10486 |
| **500** | 0.01952 |
| **750** | 0.00277 |
| **1000** | **0.00152** *(Overparameterized)* |

---

## Analysis & Conclusions

The empirical data demonstrates a fascinating machine learning phenomenon known as **Double Descent**. The performance curve can be broken down into three distinct behaviors:

1. **Underparameterized Regime ($k = 5$ to $k = 40$):**
   Initially, increasing the number of centroids drastically reduces the error. With too few units ($k < 20$), the network lacks the capacity to capture the radial, wave-like frequency of the target sine function. The local minimum error is achieved around **$k = 40$ (MSE: 0.05442)**, striking an excellent balance between computational efficiency and accuracy.
   
2. **Classical Overfitting Phase ($k = 80$ to $k = 210$):**
   As $k$ expands toward the interpolation threshold, the error begins to rise, peaking at **$k = 210$ (MSE: 0.16420)**. In this zone, the hidden units are numerous enough to start fitting local noise and sample-specific variances in the training data, degrading generalization on the unseen test set.
   
3. **Overparameterized Regime ($k \ge 500$):**
   Once $k$ surpasses the critical threshold and approaches the size of the training dataset ($k = 1000$), the error drops precipitously to a global minimum **(MSE: 0.00152)**. At this point, individual Gaussian functions are densely distributed enough to smoothly interpolate the smooth surface of the continuous function.

---
## File Structure

* **`k_means.py`**: Implementation of the custom $K$-Means clustering algorithm from scratch. Computes Euclidean distances, assigns data points to clusters, and refines centroid locations iteratively.
* **`Adaline.py`**: Implementation of the ADALINE model. Automatically injects the constant bias unit ($x_0 = 1$) into the input matrix and applies batch gradient descent using the derivative of the Sum of Squared Errors (SSE) function.
* **`RBFNN.py`**: The core neural network class. Coordinates the data mapping between the unsupervised hidden layer and the supervised output layer.
* **`test.py`**: Script responsible for data generation, dataset partitioning, running evaluation loops for various $k$ values, and plotting performance results.

---

