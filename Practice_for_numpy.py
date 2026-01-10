import numpy as np
import math 
# Generate two random 5x5 matrices
matrix_a = np.random.randint(1, 10, size=(5, 5))
matrix_b = np.random.randint(1, 10, size=(5, 5))

# Multiply the matrices
result = np.matmul(matrix_a, matrix_b)

# Display matrices and result
output = f"""
Matrix A:
{matrix_a}

Matrix B:
{matrix_b}

Result (A × B):
{result}
"""

print(output)

# ===== ESSENTIAL NUMPY FUNCTIONS FOR DATA SCIENCE =====

### 1. ARRAY CREATION
## All of array creation functions contained numpy library generates "ndarray" type variables
## It is not the same as "array" type variable, which is the primitive type used in python

print("\n=== ARRAY CREATION ===")
# Generate an n by m matrix filled with zeros
arr_zeros = np.zeros((3, 3))
print("Zeros array:\n", arr_zeros)
print("\n")

# Q) Is it possible to generate a zero matrix with NaN by NaN elements?
# A) Not possible
# array_nan = np.zeros((math.nan, math.nan))
# print("NaN array: ", array_nan)
# print("\n")

# Generate an n by m matrix filled with 1's
arr_ones = np.ones((2, 4))
print("Ones array:\n", arr_ones)
print("\n")

# Generate an n by m matrix filled with the specified number in the last argument
arr_full = np.full((3, 3), 255)
print("Full array (255):\n", arr_full)
print("\n")

arr_full = np.full(3, 196)
print("Full array (255):\n", arr_full)
print("\n")

# Generate an n-dimensional identity matrix. n is the argument of the eye() function
arr_eye= np.eye(5)
print("Identity array with 5-Dim:\n", arr_eye)
print("\n")

# Generate an array with numbers starting from "start" to "stop" with the interval "step"
# Normal ascending array
arr_range = np.arange(0, 10, 2)
print("Arange (0 to 10, step 2):", arr_range)
print("\n")

# Normal descending array
arr_range2 = np.arange(10, 0, -1)
print("Arange (10 to 0, step -1):", arr_range2)
print("\n")

# Generate numbers from 10 to 1 with ascending step 1. This is ridiculous and an empty ndarray will be generated.
arr_range3 = np.arange(10, 2)
print("Arange (10 to 2-1, step 2):", arr_range3)
print("\n")

# Generate from 10 to 14 with the default ascending step 1.
arr_range4 = np.arange(10, 15)
print("Arange (10 to 15-1, step 1):", arr_range4)
print("\n")

# Generate from 0 to 23 with the default ascending step 1.
# FYI, the default starting number for np.arange() is 0
arr_range5 = np.arange(24)
print("Arange (0 to 24-1, step 1):", arr_range5)
print("\n")

arr_linspace = np.linspace(0, 1, 5)
print("Linspace (0 to 1, 5 points):", arr_linspace)

### 2. ARRAY PROPERTIES
print("\n=== ARRAY PROPERTIES ===")

## Array Shape
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array: {arr} \n")
print(f"Shape: {arr.shape} \n")                     # returns (number of n-dim data, number of (n-1)-dim data, ... number of 1-dim data)
print(f"Type of Shape: {type(arr.shape)} \n")       # the return type of ndarray.shape is tuple

arr2 = np.array([[[1,7,3], [8,1,2]], [[1,7,3], [8,1,2]], [[1,7,3], [8,1,2]], [[1,7,3], [8,1,2]]])
print(f"Array: {arr2} \n")
print(f"Shape: {arr2.shape} \n")
print(f"Length: {len(arr2)} \n")                    # the length of an ndarray is size of the outermost dimension

# arr3 = np.array([[[1,7,3], [8,1,2]], [[1,7,3], [8,1]]]) # → Treated as an inhomogeneous shape, which leads to error
# print(f"Shape: {arr3.shape} \n")

arr4 = np.array([arr, arr])                         # Concatenating ndarrays is allowed.
print(f"Array: {arr4} \n")
print(f"Shape: {arr4.shape} \n")

for i in range(0, 20):                              # PC stopped when I tried 100 loops...
    arr = np.array([arr, arr])
print(f"Shape: {arr.shape} \n")                     

## Number of Dimension of an Array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array: {arr} \n")
print(f"Number of Dimensions: {arr.ndim} \n")       # returns the number of dimension of the ndarray. It means the number of axes of the ndarray

arr2 = np.array((1,2,3,4,5))
print(f"Array: {arr2} \n")
print(f"Number of Dimensions: {arr2.ndim} \n")       # returns the number of dimension of the ndarray. It means the number of axes of the ndarray

## Size of an Array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array: {arr} \n")
print(f"Size: {arr.size} \n")                       # returns the number of all elements contained in the ndarray -> returns 2*3 = 6

for i in range(0, 20):                              # stack arr recursively
    arr = np.array([arr, arr])
print(f"Size: {arr.size} \n")                       # returns 2^20 * 2 * 3 = 6,291,456

## Data Type of an Array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Data Type: {arr.dtype} \n")                 # returns data type of the array, which is "int64"

arr2 = np.array([[1., 2., 3.], [4., 5., 6.]])
print(f"Data Type: {arr2.dtype} \n")                # returns data type of the array, which is "float64"

arr3 = np.array([[1,8,4], [2,6,8], [7,2,3]], dtype='float') # dtype keyword can be used to designate the data type of an array when it is created
print(f"Array: {arr3} \n")                          # All the elements of the arr3 array are converted to floating point numbers
print(f"Data Type: {arr3.dtype} \n")                # the data type of arr3 is "float64" by default
arr3 = np.array([[1,8,4], [2,6,8], [7,2,3]], dtype='float32')
print(f"Data Type: {arr3.dtype} \n")                # the data type of arr3 has been casted to "float32"

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Number of Byte Chunks: {arr.itemsize} \n")  # returns length of one array element in bytes. Hard to understand.
                                                    # every element in an ndarray has the same size (by the definition of "dtype")
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype="complex128")
print(f"Number of Byte Chunks for a 128 bit complex variable: {arr.itemsize} \n")

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Total Size in Bytes: {arr.nbytes} \n")      # returns total size of an ndarray in byte

"""
NumPy axis Parameter Detailed Guide

The axis parameter is a very important concept used in NumPy to perform
operations along specific axes (dimensions) of multidimensional arrays.
"""

import numpy as np

print("=" * 70)
print("1. Understanding axis basic concepts")
print("=" * 70)

# 1D array (1 axis)
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"\n1D array:\n{arr_1d}")
print(f"shape: {arr_1d.shape}")
print(f"axis 0: The only axis of the array")

# 2D array (2 axes)
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"\n2D array:\n{arr_2d}")
print(f"shape: {arr_2d.shape}")
print(f"axis 0: Row direction ↓")
print(f"axis 1: Column direction →")

# 3D array (3 axes)
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
print(f"\n3D array:\n{arr_3d}")
print(f"shape: {arr_3d.shape}")
print(f"axis 0: Depth direction")
print(f"axis 1: Row direction")
print(f"axis 2: Column direction")

print("\n" + "=" * 70)
print("2. Using axis parameter in 2D arrays")
print("=" * 70)

# Create sample 2D array
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(f"\nOriginal array (3 rows, 4 columns):\n{matrix}")
print(f"shape: {matrix.shape}")

# axis=None (default) - operation on entire array
print(f"\nnp.sum(matrix) = {np.sum(matrix)}")
print("→ Sum of all elements: 1+2+3+...+12 = 78")

# axis=0 - operation on each column (move along row direction)
print(f"\nnp.sum(matrix, axis=0) = {np.sum(matrix, axis=0)}")
print("→ Calculate sum of each column")
print("  First column: 1+5+9 = 15")
print("  Second column: 2+6+10 = 18")
print("  Third column: 3+7+11 = 21")
print("  Fourth column: 4+8+12 = 24")
print(f"  Result shape: {np.sum(matrix, axis=0).shape}")

# axis=1 - operation on each row (move along column direction)
print(f"\nnp.sum(matrix, axis=1) = {np.sum(matrix, axis=1)}")
print("→ Calculate sum of each row")
print("  First row: 1+2+3+4 = 10")
print("  Second row: 5+6+7+8 = 26")
print("  Third row: 9+10+11+12 = 42")
print(f"  Result shape: {np.sum(matrix, axis=1).shape}")

print("\n" + "=" * 70)
print("3. Core principle for understanding axis")
print("=" * 70)

print("\nCore rule: When axis=n is specified, that axis 'disappears'")
print("\nOriginal array shape: (3, 4)")
print("After applying axis=0: (4,)  <- First dimension (3) disappears")
print("After applying axis=1: (3,)  <- Second dimension (4) disappears")

print("\nVisual understanding:")
print("\naxis=0 (↓ direction):")
print("[[1, 2, 3, 4],")
print(" [5, 6, 7, 8],")
print(" [9,10,11,12]]")
print(" ↓  ↓  ↓  ↓")
print("[15,18,21,24]")

print("\naxis=1 (→ direction):")
print("[[1, 2, 3, 4] → [10]")
print(" [5, 6, 7, 8] → [26]")
print(" [9,10,11,12] → [42]")

print("\n" + "=" * 70)
print("4. Using axis in various functions")
print("=" * 70)

data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

print(f"\nData:\n{data}\n")

# Mean
print(f"Overall mean: {np.mean(data)}")
print(f"Mean of each column (axis=0): {np.mean(data, axis=0)}")
print(f"Mean of each row (axis=1): {np.mean(data, axis=1)}")

# Max
print(f"\nOverall maximum: {np.max(data)}")
print(f"Maximum of each column (axis=0): {np.max(data, axis=0)}")
print(f"Maximum of each row (axis=1): {np.max(data, axis=1)}")

# Standard deviation
print(f"\nOverall standard deviation: {np.std(data):.2f}")
print(f"Standard deviation of each column (axis=0): {np.std(data, axis=0)}")
print(f"Standard deviation of each row (axis=1): {np.std(data, axis=1)}")

print("\n" + "=" * 70)
print("5. Using axis in 3D arrays")
print("=" * 70)

# 3D array: (2, 3, 4) - 2 layers, each layer is 3 rows x 4 columns
cube = np.array([[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]],
                 
                 [[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]]])

print(f"\n3D array shape: {cube.shape}")
print(f"First layer:\n{cube[0]}")
print(f"\nSecond layer:\n{cube[1]}")

print(f"\nnp.sum(cube, axis=0).shape: {np.sum(cube, axis=0).shape}")
print("→ Sum along layers (2 layers → one)")
print(f"Result:\n{np.sum(cube, axis=0)}")

print(f"\nnp.sum(cube, axis=1).shape: {np.sum(cube, axis=1).shape}")
print("→ Sum along rows in each layer (3 rows → one)")
print(f"Result:\n{np.sum(cube, axis=1)}")

print(f"\nnp.sum(cube, axis=2).shape: {np.sum(cube, axis=2).shape}")
print("→ Sum along columns in each layer and row (4 columns → one)")
print(f"Result:\n{np.sum(cube, axis=2)}")

print("\n" + "=" * 70)
print("6. Specifying multiple axes simultaneously")
print("=" * 70)

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"\nOriginal array:\n{matrix}")
print(f"shape: {matrix.shape}")

# Specify multiple axes (as tuple)
print(f"\nnp.sum(matrix, axis=(0, 1)): {np.sum(matrix, axis=(0, 1))}")
print("→ Sum across all axes (same as axis=None)")

# Specify multiple axes in 3D array
print(f"\nnp.sum(cube, axis=(0, 2)).shape: {np.sum(cube, axis=(0, 2)).shape}")
print(f"Result: {np.sum(cube, axis=(0, 2))}")
print("→ Axes 0 and 2 disappear, only axis 1 remains")

print("\n" + "=" * 70)
print("7. Using keepdims parameter")
print("=" * 70)

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(f"\nOriginal array:\n{arr}")
print(f"shape: {arr.shape}")

result_normal = np.sum(arr, axis=1)
print(f"\nnp.sum(arr, axis=1):")
print(f"Result: {result_normal}")
print(f"shape: {result_normal.shape}")

result_keepdims = np.sum(arr, axis=1, keepdims=True)
print(f"\nnp.sum(arr, axis=1, keepdims=True):")
print(f"Result:\n{result_keepdims}")
print(f"shape: {result_keepdims.shape}")
print("→ Maintains dimension, useful for broadcasting")

print("\n" + "=" * 70)
print("8. Practical example: Student grade data analysis")
print("=" * 70)

# 3 students, 4 subjects (Math, English, Science, History)
scores = np.array([[85, 90, 78, 92],  # Student 1
                   [88, 85, 91, 87],  # Student 2
                   [92, 88, 85, 90]]) # Student 3

subjects = ['Math', 'English', 'Science', 'History']
students = ['Student 1', 'Student 2', 'Student 3']

print("\nGrade Table:")
print(f"{'':12}", end='')
for subject in subjects:
    print(f"{subject:>10}", end='')
print()

for i, student in enumerate(students):
    print(f"{student:12}", end='')
    for score in scores[i]:
        print(f"{score:10}", end='')
    print()

# Average by subject (axis=0)
subject_avg = np.mean(scores, axis=0)
print("\nAverage score by subject:")
for subject, avg in zip(subjects, subject_avg):
    print(f"{subject}: {avg:.1f} points")

# Average by student (axis=1)
student_avg = np.mean(scores, axis=1)
print("\nAverage score by student:")
for student, avg in zip(students, student_avg):
    print(f"{student}: {avg:.1f} points")

# Maximum by subject (axis=0)
subject_max = np.max(scores, axis=0)
print("\nHighest score by subject:")
for subject, max_score in zip(subjects, subject_max):
    print(f"{subject}: {max_score} points")

# Maximum by student (axis=1)
student_max = np.max(scores, axis=1)
print("\nHighest score by student:")
for student, max_score in zip(students, student_max):
    print(f"{student}: {max_score} points")

print("\n" + "=" * 70)
print("9. Cumulative operations with axis")
print("=" * 70)

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(f"\nOriginal array:\n{arr}")

print(f"\nnp.cumsum(arr, axis=0) (cumulative sum along rows):")
print(np.cumsum(arr, axis=0))
print("→ Cumulative sum from top to bottom in each column")

print(f"\nnp.cumsum(arr, axis=1) (cumulative sum along columns):")
print(np.cumsum(arr, axis=1))
print("→ Cumulative sum from left to right in each row")

print("\n" + "=" * 70)
print("Summary and Key Points to Remember")
print("=" * 70)

print("""
1. axis specifies the 'direction' for performing operations
   - axis=0: Row direction (↓)
   - axis=1: Column direction (→)
   
2. When axis is specified, that dimension is 'reduced'
   - (3, 4) array with axis=0 applied → (4,) result
   - (3, 4) array with axis=1 applied → (3,) result
   
3. axis=None (default) performs operation on entire array
   
4. Multiple axes can be specified simultaneously: axis=(0, 1)
   
5. Dimension can be maintained with keepdims=True
   
6. Intuitive understanding: "if axis=n, repeat along the nth index"
   - axis=0: Access in form arr[i, :] and perform operation
   - axis=1: Access in form arr[:, j] and perform operation
""")
print("\n" + "=" * 70)
print("NumPy axis Parameter Detailed Guide")
print("=" * 70)


### 3. MATHEMATICAL OPERATIONS
print("\n=== MATHEMATICAL OPERATIONS ===")
arr = np.array([[1,6,2], [8,2,3], [5,9,1]])
arr3D = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])
arr4D = np.array([[[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]], [[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]], [[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]]])

# Sum of an ndarray
print("Sum:", np.sum(arr))                          # Sum of all elements in the array
print("\n")
print("Sum along 0th axis:", np.sum(arr, 0))        # Column-wise sum
print("\n")
print("Sum along 0th axis:", arr.sum(axis=0))       # sum function seems to be a member function of a ndarray class
print("\n")
print("Sum along 1st axis:", np.sum(arr, 1))        # Row-wise sum
print("\n")
print("Sum along 1st axis:", arr.sum(axis=1))
print("\n")
# print("Sum along 2nd axis:", arr.sum(axis=2))       # error. arr is 2-dim array and, therefore, accessing axis 2 is not allowed
# print("\n")
print("Sum of the 3D array along 0th axis:", arr3D.sum(axis=0))
print("\n")
print("Sum of the 3D array along 1th axis:", arr3D.sum(axis=1))
print("\n")
print("Sum of the 3D array along 2th axis:", arr3D.sum(axis=2))
print("\n")                                         # I got the idea for the concept of axis in ndarray
print("Sum of the 3D array along 0th axis:", arr4D.sum(axis=0))
print("\n")
print("Sum of the 4D array along 1th axis:", arr4D.sum(axis=1))
print("\n")
print("Sum of the 4D array along 1th axis:", arr4D.sum(axis=2))
print("\n")
print("Sum of the 4D array along 2th axis:", arr4D.sum(axis=3))
print("\n")                                         # I got the idea for the concept of axis in ndarray

# mean of an ndarray
print("Mean:", np.mean(arr))                        # Average value of the whole element of the array
print("\n")
print("Mean along 0th axis:", arr.mean(axis=0))     # The mean function also uses the concept of axis. It is the same as the sum function
print("\n")
print("Mean along 1st axis:", arr.mean(axis=1))     # 
print("\n")
print("Mean of the 3D array along 0th axis:", arr3D.mean(axis=0))
print("\n")
print("Mean of the 3D array along 1th axis:", arr3D.mean(axis=1))
print("\n")
print("Mean of the 3D array along 2th axis:", arr3D.mean(axis=2))
print("\n")                                         

# Standard deviation of an ndarray
print("Std Dev:", np.std(arr))
print("\n")
print("Standard deviation along 0th axis:", arr.std(axis=0))     # Very similar to mean function
print("\n")
print("Standard deviation along 1st axis:", arr.std(axis=1))
print("\n")
print("Standard deviation of the 3D array along 0th axis:", arr3D.std(axis=0))
print("\n")
print("Standard deviation of the 3D array along 1th axis:", arr3D.std(axis=1))
print("\n")
print("Standard deviation of the 3D array along 2th axis:", arr3D.std(axis=2))
print("\n")

# min and max function of ndarray
print("Min:", np.min(arr), "Max:", np.max(arr))
print("\n")
print("Min:", np.min(arr3D), "Max:", np.max(arr3D))
print("\n")
print("Min:", np.min(arr3D, axis=0), "\nMax:", np.max(arr3D, axis=0))
print("\n")
print("Min:", np.min(arr3D, axis=1), "\nMax:", np.max(arr3D, axis=1))
print("\n")
print("Min:", np.min(arr3D, axis=2), "\nMax:", np.max(arr3D, axis=2))
print("\n")
print("Min:", np.min(arr3D, axis=(1,2)), "\nMax:", np.max(arr3D, axis=(1,2)))   # Interesting... finding maximum along the axis 1 and than do it again along the axis 2
print("\n")

# median function of Numpy (Not a method of ndarray object)
print("Median:", np.median(arr))
print("\n")
print("Median along 0th axis:", np.median(arr, axis=0))     # Numpy.ndarray object does not have method named "median"
print("\n")
print("Median along 1st axis:", np.median(arr, axis=1))     # median is a member function of Numpy
print("\n")
print("Median of the 3D array along 0th axis:", np.median(arr3D, axis=0))
print("\n")
print("Median of the 3D array along 1th axis:", np.median(arr3D, axis=1))
print("\n")
print("Median of the 3D array along 2th axis:", np.median(arr3D, axis=2))
print("\n")    

# variance function of an ndarray
print("Var:", np.var(arr))
print("\n")
print("Variance along 0th axis:", arr.var(axis=0))
print("\n")
print("Variance along 1st axis:", arr.var(axis=1))
print("\n")
print("Variance of the 3D array along 0th axis:", arr3D.var(axis=0))
print("\n")
print("Variance of the 3D array along 1th axis:", arr3D.var(axis=1))
print("\n")
print("Variance of the 3D array along 2th axis:", arr3D.var(axis=2))
print("\n")

arr1D = np.array([1, 2, 3, 4])
arr2D = np.array([[1,2],[3,4]])
arr3D = np.array([[[1,2],[3,4]], [[1,2],[3,4]]])
print("Product:", np.prod(arr1D))
print("\n")
print("Col.-wise product: ", np.prod(arr2D, axis=0))
print("\n")
print("Row.-wise product: ", np.prod(arr2D, axis=1))
print("\n")

# print("Cumulative sum:", np.cumsum(arr))
# print("\n")
# print("Cumulative product:", np.cumprod(arr))
# print("\n")
# print("Absolute values:", np.abs(np.array([[-1, 2], [-3, 4]])))
# print("\n")
# print("Square root:", np.sqrt(arr.astype(float)))
# print("\n")
# print("Power (squared):", np.power(arr, 2))
# print("\n")
# print("Natural log:", np.log(arr.astype(float)))
# print("\n")
# print("Base 10 log:", np.log10(arr.astype(float)))
# print("\n")
# print("Exponential:", np.exp(np.array([0, 1, 2])))
# print("\n")
# print("Trigonometric sin:", np.sin(np.array([0, np.pi/2, np.pi])))
# print("\n")
# print("Trigonometric cos:", np.cos(np.array([0, np.pi/2, np.pi])))
# print("\n")
# print("Round:", np.round(np.array([1.2, 2.7, 3.5])))
# print("\n")
# print("Floor:", np.floor(np.array([1.2, 2.7, 3.5])))
# print("\n")
# print("Ceil:", np.ceil(np.array([1.2, 2.7, 3.5])))
# print("\n")
# print("Percentile 75th:", np.percentile(arr, 75))
# print("\n")
# print("Quantile 0.5 (median):", np.quantile(arr, 0.5))
# print("\n")
# print("Dot product:", np.dot(arr[0], arr[1]))
# print("\n")
# print("Cross product:", np.cross([1, 2, 3], [4, 5, 6]))
# print("\n")
# print("Outer product:", np.outer(arr[0], arr[1]))
# print("\n")
# print("Inner product:", np.inner(arr[0], arr[1]))
# print("\n")
# print("Matrix rank:", np.linalg.matrix_rank(arr))
# print("\n")
# print("Determinant:", np.linalg.det(arr.astype(float)))
# print("\n")
# print("Trace:", np.trace(arr))
# print("\n")
# print("Eigenvalues:", np.linalg.eigvals(arr.astype(float)))
# print("\n")
# print("Norm (L2):", np.linalg.norm(arr))
# print("\n")
# print("Norm (L1):", np.linalg.norm(arr, ord=1))
# print("\n")
# print("QR decomposition:", np.linalg.qr(arr.astype(float)))
# print("\n")
# print("SVD:", np.linalg.svd(arr.astype(float)))
# print("\n")
# print("Condition number:", np.linalg.cond(arr.astype(float)))
# print("\n")
# print("Matrix inverse:", np.linalg.inv(arr.astype(float)))
# print("\n")

# # 4. RESHAPING & FLATTENING
# print("\n=== RESHAPING & FLATTENING ===")
# reshaped = arr.reshape(3, 2)
# print("Reshaped:\n", reshaped)
# print("Flattened:", arr.flatten())

# # 5. INDEXING & SLICING
# print("\n=== INDEXING & SLICING ===")
# print("First row:", arr[0])
# print("Element [1,2]:", arr[1, 2])
# print("Slice [0:2, 1:3]:\n", arr[0:2, 1:3])

# # 6. CONCATENATION & STACKING
# print("\n=== CONCATENATION & STACKING ===")
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# print("Concatenate:", np.concatenate([arr1, arr2]))
# print("Stack:\n", np.stack([arr1, arr2]))

# # 7. SORTING & SEARCHING
# print("\n=== SORTING & SEARCHING ===")
# unsorted = np.array([3, 1, 4, 1, 5, 9])
# print("Sorted:", np.sort(unsorted))
# print("Argsort (indices):", np.argsort(unsorted))

# # 8. LOGICAL OPERATIONS
# print("\n=== LOGICAL OPERATIONS ===")
# arr_data = np.array([1, 2, 3, 4, 5])
# print("Values > 3:", arr_data[arr_data > 3])
# print("Where (replace > 3 with 0):\n", np.where(arr_data > 3, 0, arr_data))

# # 9. STATISTICAL FUNCTIONS
# print("\n=== STATISTICAL FUNCTIONS ===")
# data = np.array([10, 20, 30, 40, 50])
# print("Percentile 75:", np.percentile(data, 75))
# print("Median:", np.median(data))
# print("Variance:", np.var(data))

# # 10. RANDOM SAMPLING
# print("\n=== RANDOM SAMPLING ===")
# print("Random floats:", np.random.rand(3))
# print("Random integers:", np.random.randint(1, 100, 5))
# print("Normal distribution:", np.random.normal(0, 1, 3))

# # 11. ADVANCED ARRAY OPERATIONS
# print("\n=== ADVANCED ARRAY OPERATIONS ===")
# arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print("Transpose:\n", arr_2d.T)
# print("Diagonal:", np.diag(arr_2d))
# print("Trace (sum of diagonal):", np.trace(arr_2d))

# # 12. MATRIX OPERATIONS
# print("\n=== MATRIX OPERATIONS ===")
# mat1 = np.array([[1, 2], [3, 4]])
# mat2 = np.array([[5, 6], [7, 8]])
# print("Matrix multiplication:\n", np.dot(mat1, mat2))
# print("Element-wise multiplication:\n", mat1 * mat2)
# print("Determinant:", np.linalg.det(mat1))
# print("Inverse:\n", np.linalg.inv(mat1))

# # 13. EIGENVALUES & EIGENVECTORS
# print("\n=== EIGENVALUES & EIGENVECTORS ===")
# eigenvalues, eigenvectors = np.linalg.eig(mat1)
# print("Eigenvalues:", eigenvalues)
# print("Eigenvectors:\n", eigenvectors)

# # 14. BROADCASTING
# print("\n=== BROADCASTING ===")
# arr_a = np.array([[1, 2, 3], [4, 5, 6]])
# scalar = 10
# print("Array + Scalar:\n", arr_a + scalar)
# arr_b = np.array([1, 2, 3])
# print("2D + 1D (broadcasting):\n", arr_a + arr_b)

# # 15. APPLY FUNCTIONS
# print("\n=== APPLY FUNCTIONS ===")
# arr_func = np.array([1, 2, 3, 4, 5])
# print("Square root:", np.sqrt(arr_func))
# print("Exponential:", np.exp(arr_func))
# print("Log:", np.log(arr_func))

# # 16. AGGREGATION ALONG AXES
# print("\n=== AGGREGATION ALONG AXES ===")
# arr_axis = np.array([[1, 2, 3], [4, 5, 6]])
# print("Sum axis=0 (columns):", np.sum(arr_axis, axis=0))
# print("Sum axis=1 (rows):", np.sum(arr_axis, axis=1))
# print("Mean along axis=0:", np.mean(arr_axis, axis=0))

# # 17. UNIQUE & COUNTS
# print("\n=== UNIQUE & COUNTS ===")
# arr_unique = np.array([1, 2, 2, 3, 3, 3, 4])
# unique_vals, counts = np.unique(arr_unique, return_counts=True)
# print("Unique values:", unique_vals)
# print("Counts:", counts)

# # 18. COMPARISON & MASKING
# print("\n=== COMPARISON & MASKING ===")
# arr_compare = np.array([1, 5, 3, 8, 2])
# mask = arr_compare > 3
# print("Mask (> 3):", mask)
# print("Masked values:", arr_compare[mask])
# print("Count values > 3:", np.sum(arr_compare > 3))

# # 19. SET OPERATIONS
# print("\n=== SET OPERATIONS ===")
# set1 = np.array([1, 2, 3, 4])
# set2 = np.array([3, 4, 5, 6])
# print("Intersection:", np.intersect1d(set1, set2))
# print("Union:", np.union1d(set1, set2))
# print("Difference:", np.setdiff1d(set1, set2))

# # 20. ADVANCED RANDOM SAMPLING
# print("\n=== ADVANCED RANDOM SAMPLING ===")
# print("Choice (random selection):", np.random.choice([1, 2, 3, 4, 5], size=3, replace=False))
# print("Shuffle:", np.random.shuffle(arr_unique), arr_unique)
# print("Poisson distribution:", np.random.poisson(lam=3, size=5))

# # 21. LINEAR ALGEBRA (np.linalg) - COMPREHENSIVE GUIDE
# print("\n" + "="*60)
# print("=== COMPREHENSIVE LINEAR ALGEBRA (np.linalg) GUIDE ===")
# print("="*60)

# # Basic matrices for demonstration
# A = np.array([[4, 7], [2, 6]])
# B = np.array([[1, 2], [3, 4]])
# C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# v = np.array([1, 2, 3])

# # 21.1 NORM (Vector and Matrix Norms)
# print("\n--- NORM ---")
# print("Vector norm (L2):", np.linalg.norm(v))
# print("Vector norm (L1):", np.linalg.norm(v, ord=1))
# print("Matrix Frobenius norm:", np.linalg.norm(A))

# # 21.2 DETERMINANT
# print("\n--- DETERMINANT ---")
# print("det(A):", np.linalg.det(A))
# print("det(B):", np.linalg.det(B))

# # 21.3 MATRIX RANK
# print("\n--- MATRIX RANK ---")
# print("rank(A):", np.linalg.matrix_rank(A))
# print("rank(C):", np.linalg.matrix_rank(C))

# # 21.4 INVERSE
# print("\n--- MATRIX INVERSE ---")
# A_inv = np.linalg.inv(A)
# print("A^(-1):\n", A_inv)
# print("Verify A × A^(-1):\n", np.dot(A, A_inv))

# # 21.5 TRACE
# print("\n--- TRACE (Sum of Diagonal) ---")
# print("trace(A):", np.trace(A))
# print("trace(B):", np.trace(B))

# # 21.6 TRANSPOSE
# print("\n--- TRANSPOSE ---")
# print("A.T:\n", A.T)
# print("C.T:\n", C.T)

# # 21.7 EIGENVALUES & EIGENVECTORS
# print("\n--- EIGENVALUES & EIGENVECTORS ---")
# eigenvals, eigenvecs = np.linalg.eig(A)
# print("Eigenvalues:", eigenvals)
# print("Eigenvectors:\n", eigenvecs)

# # 21.8 QR DECOMPOSITION
# print("\n--- QR DECOMPOSITION (A = Q×R) ---")
# Q, R = np.linalg.qr(B)
# print("Q:\n", Q)
# print("R:\n", R)
# print("Verify Q×R:\n", np.dot(Q, R))

# # 21.9 SVD (Singular Value Decomposition)
# print("\n--- SINGULAR VALUE DECOMPOSITION (A = U×Σ×V^T) ---")
# U, S, VT = np.linalg.svd(C)
# print("U:\n", U)
# print("Singular values:", S)
# print("V^T:\n", VT)

# # 21.10 CHOLESKY DECOMPOSITION
# print("\n--- CHOLESKY DECOMPOSITION ---")
# pos_def = np.array([[4, 2], [2, 3]])
# L = np.linalg.cholesky(pos_def)
# print("L:\n", L)
# print("Verify L×L^T:\n", np.dot(L, L.T))

# # 21.11 SOLVING LINEAR SYSTEMS (Ax = b)
# print("\n--- SOLVING LINEAR SYSTEMS ---")
# b = np.array([5, 11])
# x = np.linalg.solve(A, b)
# print("Solution to Ax=b:", x)
# print("Verify A×x=b:", np.dot(A, x))

# # 21.12 LEAST SQUARES SOLUTION
# print("\n--- LEAST SQUARES SOLUTION ---")
# A_rect = np.array([[1, 1], [1, 2], [1, 3]])
# b_rect = np.array([2, 3, 5])
# x_lstsq, residuals, rank, s = np.linalg.lstsq(A_rect, b_rect, rcond=None)
# print("Least squares solution:", x_lstsq)
# print("Residuals:", residuals)

# # 21.13 MATRIX POWER
# print("\n--- MATRIX POWER ---")
# A_squared = np.linalg.matrix_power(A, 2)
# print("A^2:\n", A_squared)
# A_cubed = np.linalg.matrix_power(A, 3)
# print("A^3:\n", A_cubed)

# # 21.14 CONDITION NUMBER
# print("\n--- CONDITION NUMBER ---")
# cond = np.linalg.cond(A)
# print("Condition number of A:", cond)

# # 21.15 PSEUDO-INVERSE (Moore-Penrose)
# print("\n--- PSEUDO-INVERSE ---")
# C_pinv = np.linalg.pinv(C)
# print("Pseudo-inverse of C:\n", C_pinv)

# # 21.16 MATRIX OPERATIONS CHAINING
# print("\n--- MATRIX OPERATIONS CHAINING ---")
# result = np.linalg.multi_dot([A, B, A.T])
# print("A × B × A^T:\n", result)