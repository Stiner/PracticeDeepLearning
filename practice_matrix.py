import numpy as np

def main():
    mat1 = np.array(
        [
            [2],
            [1],
            [6]
        ]
    )

    mat2 = np.array(
        [
            [1,3,5,6,1,4,7,8,9,2,4,5]
        ]
    )

    print(mat1.shape)
    print(mat2.shape)

    result = np.dot(mat1, mat2)
    print(result.shape)
    print(result)