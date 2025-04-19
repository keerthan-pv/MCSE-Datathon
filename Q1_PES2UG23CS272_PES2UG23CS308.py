import numpy as np
import sys

class DimensionalityReducer:
    def _init_(self):
        self.mean = None
        self.basis = None  # Principal components (rows)
        self.variance_explained = None

    def preprocess(self, X):
        self.mean=np.mean(X,axis=0)
        return X-self.mean

    def compute_key_directions(self, X_centered):
        a,b,Vt=np.linalg.svd(X_centered, full_matrices=False)
        self.basis=Vt
        total_variance=np.sum(b**2)
        self.variance_explained=(b**2)/total_variance if total_variance >0 else np.zeros_like(b)

       
    def reduce_dimensions(self, X_centered, k):
        return X_centered@self.basis[:k].T

    def reconstruct(self, X_reduced):
        ##HERE, FILL IN THE BLANKS
        if X_reduced.size == 0 :
            return np.empty_like(self.mean)
        return X_reduced @ self.basis[:X_reduced.shape[1]]+self.mean
        

    def evaluate_error(self, X_original, X_reconstructed):
        return np.linalg.norm(X_original-X_reconstructed)
        

def main():
    # Read input matrix from stdin
    A = []
    while True:
        try:
            row = input().strip()
            if row:
                A.append(list(map(float, row.split())))
            else:
                break 
        except EOFError:
            break
    A = np.array(A)
    
    reducer = DimensionalityReducer()
    
    # Step 1: Preprocess (center the data)
    try:
        X_centered = reducer.preprocess(A)
    except Exception as e:
        print("Error during preprocessing:", e)
        return
    
    print("Centered data:")
    print(X_centered)
    
    # Step 2: Compute key directions (SVD)
    try:
        reducer.compute_key_directions(X_centered)
    except Exception as e:
        print("Error during key directions computation:", e)
        return
    
    # Determine k for 95% variance
    cumulative_variance = np.cumsum(reducer.variance_explained)
    k = np.argmax(cumulative_variance >= 0.95) + 1
    if k == 0:  # Handle case where no component meets the threshold
        k = len(reducer.variance_explained)
    
    # Handle case when all variances are zero (e.g., constant features)
    if np.allclose(X_centered, 0):
        k = 0
    
    # Step 3: Reduce dimensions
    try:
        if k > 0:
            X_reduced = reducer.reduce_dimensions(X_centered, k)
        else:
            X_reduced = np.zeros((X_centered.shape[0], 0))
    except Exception as e:
        print("Error during dimensionality reduction:", e)
        return
    
    print("\nTop directions:")
    if k > 0:
        for direction in reducer.basis[:k]:
            print(" ".join(f"{x:.2f}" for x in direction))
    else:
        print("No directions (all features are constant)")
    
    print("\nReduced data:")
    if X_reduced.size > 0:
        for row in X_reduced:
            print(" ".join(f"{x:.2f}" for x in row))
    else:
        print("No reduced data (all features are constant)")
    
    # Step 4: Reconstruct data
    try:
        X_reconstructed = reducer.reconstruct(X_reduced)
    except Exception as e:
        print("Error during reconstruction:", e)
        return
    
    print("\nReconstructed data:")
    print(X_reconstructed)
    
    # Step 5: Evaluate reconstruction error
    try:
        error = reducer.evaluate_error(A, X_reconstructed)
    except Exception as e:
        print("Error during error evaluation:", e)
        return
    
    print(f"\nReconstruction error: {error:.2f}")

if __name__ == "__main__":
    main()