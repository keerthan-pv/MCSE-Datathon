import numpy as np

def find_fundamental_subspaces(matrix):
    """
    Find the dimensions of column space and null space of a given matrix.
    
    Args:
        matrix: A 2D numpy array
    
    Returns:
        A dictionary with the dimensions of column space and null space
    """
    matrix=np.array(matrix) 
    m,n=matrix.shape 
    rank=np.linalg.matrix_rank(matrix) 
    column_space_dim=rank 
    null_space_dim=n-rank
    result={ }
    result['column_space_dim']=column_space_dim
    result['null_space_dim']=null_space_dim 
    return result

if __name__ == "__main__":
    # Get matrix dimensions from user
    m, n = map(int, input().strip().split())
    
    # Get matrix entries
    matrix = []
    for i in range(m):
        row = list(map(float, input().strip().split()))
        if len(row) != n:
            print(f"Error: Expected {n} elements in row {i+1}, but got {len(row)}")
            exit(1)
        matrix.append(row)
    
    result = find_fundamental_subspaces(matrix)
    
    print(result['column_space_dim'], result['null_space_dim'])