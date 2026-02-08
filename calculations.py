import numpy as np


def clean_val(v):
    """
    Format a number to be an int if it's close to an int.
    Handles floats and complex numbers (if imag part is negligible).
    """
    # 1. Handle Complex -> Real if imaginary part is zero
    if np.iscomplexobj(v) or isinstance(v, complex):
        if np.isclose(v.imag, 0, atol=1e-8):
            v = v.real
        else:
            return np.round(v, 2) # Return rounded complex
            
    # 2. Handle Float -> Int
    # Check if value is essentially an integer
    if np.isclose(v, np.round(v), atol=1e-8):
        return int(np.round(v))
    
    # 3. Default float rounding
    return round(float(v), 3)

def get_rref(matrix):
    """Computes RREF using Gaussian elimination."""
    M = matrix.copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r >= rows: break
        
        pivot = np.argmax(np.abs(M[r:, c])) + r
        
        if np.isclose(M[pivot, c], 0):
            continue
            
        M[[r, pivot]] = M[[pivot, r]]
        
        # Scale pivot to 1
        M[r] = M[r] / M[r, c]
        
        # Eliminate column
        for i in range(rows):
            if i != r:
                M[i] -= M[i, c] * M[r]
        r += 1
        
    M[np.abs(M) < 1e-10] = 0
    return M

def analyze_matrix(matrix: np.ndarray) -> dict:
    """Calculates various properties of the matrix."""
    results = {}
    is_square = matrix.shape[0] == matrix.shape[1]
    results["is_square"] = is_square
    results["shape"] = matrix.shape
    results["shape_str"] = f"{matrix.shape[0]}×{matrix.shape[1]}"
    results["rows"] = matrix.shape[0]
    results["cols"] = matrix.shape[1]
    
    # 1. Determinant
    if is_square:
        det = np.linalg.det(matrix)
        results['determinant'] = clean_val(det)
    else:
        results['determinant'] = None

    # 2. Rank
    results['rank'] = np.linalg.matrix_rank(matrix)

    # 3. Inverse
    if is_square:
        try:
            inv = np.linalg.inv(matrix)
            results['inverse'] = np.round(inv, 4)
        except np.linalg.LinAlgError:
            results['inverse'] = "Undefined (Singular Matrix)"
    else:
        results['inverse'] = None

    # 4. Eigenvalues & Vectors

    if is_square:
        eig_vals, eig_vectors = np.linalg.eig(matrix)
        
        # Sort descending
        idx = eig_vals.argsort()[::-1]   
        eig_vals = eig_vals[idx]
        eig_vectors = eig_vectors[:,idx]

        grouped_data = []
        for i in range(len(eig_vals)):
            current_val = eig_vals[i]
            current_vec = eig_vectors[:, i]
            
            found_group = None
            for group in grouped_data:
                if np.isclose(current_val, group['raw_val'], atol=1e-5):
                    found_group = group
                    break
            
            if found_group:
                found_group['vectors'].append(current_vec)
                found_group['dim'] += 1
            else:
                grouped_data.append({
                    'raw_val': current_val,
                    'vectors': [current_vec],
                    'dim': 1
                })

        dim_eigenraum_1 = 0
        for group in grouped_data:
            if np.isclose(group['raw_val'], 1, atol=1e-5):
                dim_eigenraum_1 = group['dim']
                break
        results['eigenraum_dim_1'] = dim_eigenraum_1

        eigen_data = []
        for group in grouped_data:
            val_display = clean_val(group['raw_val'])
            formatted_vectors = []
            for vec in group['vectors']:
                # Clean each component of the vector
                cleaned_vec = [clean_val(x) for x in vec]
                formatted_vectors.append(cleaned_vec)
                
            eigen_data.append({
                "value": val_display,
                "vectors": formatted_vectors,
                "dim": group["dim"]
            })

        results['eigen_data'] = eigen_data
    else:
        results['eigen_data'] = []
        results['eigenraum_dim_1'] = 0

    # 5. Trace
    results['trace'] = clean_val(np.trace(matrix)) if is_square else None

    # 6. Transpose
    results['trans'] = np.transpose(matrix)

    # 7. RREF
    results['rref'] = np.round(get_rref(matrix), 2)



    # 8 Orthogonale Abbildung check
    results['orth'] = (
        np.allclose(results['trans'] @ matrix, np.eye(matrix.shape[0]))
        if is_square else None
    )

    results['abb_type'] = None
    if is_square and results['orth']:
        det = np.linalg.det(matrix)
        tr = np.trace(matrix)
        if matrix.shape == (2, 2):
            if np.isclose(det, 1.0, atol=1e-5):
                if np.isclose(tr, 2.0, atol=1e-5):
                    results['abb_type'] = "Drehung mit Winkel 0"
                elif np.isclose(tr, -2.0, atol=1e-5):
                    results['abb_type'] = "Drehung mit Winkel π"
                else:
                    results['abb_type'] = "Drehung mit Winkel != 0"
            elif np.isclose(det, -1.0, atol=1e-5):
                results['abb_type'] = "Spiegelung"
        elif matrix.shape == (3, 3):
            
                if dim_eigenraum_1 == 3:
                    results['abb_type'] = "Drehung (3D) mit Winkel 0"
                elif dim_eigenraum_1 == 1:
                    results['abb_type'] = "Drehung (3D) mit Winkel != 0"
                elif dim_eigenraum_1 == 2:
                    results['abb_type'] = "Spieglung"
                else:
                    results['abb_type'] = "Drehspiegelung (3D)"

    return results