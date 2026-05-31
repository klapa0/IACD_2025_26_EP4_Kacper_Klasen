import numpy as np
import matplotlib.pyplot as plt

from RBFNN import RBFNN

def generate_data(n_samples=1000):
    """
    Generating collection from  f(x,y) = sin(sqrt(x^2 + y^2))
    for x: [-5, 5], y: [-5, 5].
    """
    # np.random.seed(42)
    
    X = np.random.uniform(-5, 5, size=(n_samples, 2))
    
    y = np.sin(np.sqrt(X[:, 0]**2 + X[:, 1]**2))
    
    return X, y

def run_experiment():
    X_train, y_train = generate_data(1500)
    X_test, y_test = generate_data(500)
    
    # Testing data
    hidden_units_list = [5, 10, 20, 40, 80, 100, 120 , 150, 180, 210, 250, 500, 750, 1000]
    final_mses = []
    
    for k in hidden_units_list:
        model = RBFNN(num_hidden_units=k, eta=0.1, epochs=1500)
        model.fit(X_train, y_train)
        
        # Calculating MSE for prediction
        predictions = model.predict(X_test)
        mse = np.mean((y_test - predictions)**2)
        final_mses.append(mse)
        
        print(f"Centroids (k): {k:3d} | (MSE): {mse:.5f}")
    
    # Showing plot
    plt.figure(figsize=(9, 5))
    plt.plot(hidden_units_list, final_mses, marker='o', linestyle='-', color='teal', linewidth=2)
    plt.title('Relation between MSE and numbers of centroids (RBFNN)', fontsize=14)
    plt.xlabel('Number of units in RBF (k)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("RBF_K_vs_MSE.png")
    plt.show()


run_experiment()