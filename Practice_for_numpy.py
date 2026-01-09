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

# Generate numbers from 10 to 1 with ascending step 1. This is rediculous and an empty ndarray will be generated.
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
NumPy axis 파라미터 상세 가이드

axis 파라미터는 NumPy에서 다차원 배열의 특정 축(dimension)을 따라
연산을 수행할 때 사용하는 매우 중요한 개념입니다.
"""

import numpy as np

print("=" * 70)
print("1. axis 기본 개념 이해하기")
print("=" * 70)

# 1D 배열 (축이 1개)
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"\n1D 배열:\n{arr_1d}")
print(f"shape: {arr_1d.shape}")
print(f"axis 0: 배열의 유일한 축")

# 2D 배열 (축이 2개)
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"\n2D 배열:\n{arr_2d}")
print(f"shape: {arr_2d.shape}")
print(f"axis 0: 행(row) 방향 ↓")
print(f"axis 1: 열(column) 방향 →")

# 3D 배열 (축이 3개)
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
print(f"\n3D 배열:\n{arr_3d}")
print(f"shape: {arr_3d.shape}")
print(f"axis 0: 깊이(depth) 방향")
print(f"axis 1: 행(row) 방향")
print(f"axis 2: 열(column) 방향")

print("\n" + "=" * 70)
print("2. 2D 배열에서 axis 파라미터 사용 예시")
print("=" * 70)

# 샘플 2D 배열 생성
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(f"\n원본 배열 (3행 4열):\n{matrix}")
print(f"shape: {matrix.shape}")

# axis=None (기본값) - 전체 배열에 대해 연산
print(f"\nnp.sum(matrix) = {np.sum(matrix)}")
print("→ 모든 원소의 합: 1+2+3+...+12 = 78")

# axis=0 - 각 열에 대해 연산 (행 방향으로 이동하며 계산)
print(f"\nnp.sum(matrix, axis=0) = {np.sum(matrix, axis=0)}")
print("→ 각 열의 합을 계산")
print("  첫 번째 열: 1+5+9 = 15")
print("  두 번째 열: 2+6+10 = 18")
print("  세 번째 열: 3+7+11 = 21")
print("  네 번째 열: 4+8+12 = 24")
print(f"  결과 shape: {np.sum(matrix, axis=0).shape}")

# axis=1 - 각 행에 대해 연산 (열 방향으로 이동하며 계산)
print(f"\nnp.sum(matrix, axis=1) = {np.sum(matrix, axis=1)}")
print("→ 각 행의 합을 계산")
print("  첫 번째 행: 1+2+3+4 = 10")
print("  두 번째 행: 5+6+7+8 = 26")
print("  세 번째 행: 9+10+11+12 = 42")
print(f"  결과 shape: {np.sum(matrix, axis=1).shape}")

print("\n" + "=" * 70)
print("3. axis를 이해하는 핵심 원리")
print("=" * 70)

print("\n핵심 규칙: axis=n을 지정하면 해당 축이 '사라집니다'")
print("\n원본 배열 shape: (3, 4)")
print("axis=0 적용 후: (4,)  <- 첫 번째 차원(3)이 사라짐")
print("axis=1 적용 후: (3,)  <- 두 번째 차원(4)이 사라짐")

print("\n시각적 이해:")
print("\naxis=0 (↓ 방향):")
print("[[1, 2, 3, 4],")
print(" [5, 6, 7, 8],")
print(" [9,10,11,12]]")
print(" ↓  ↓  ↓  ↓")
print("[15,18,21,24]")

print("\naxis=1 (→ 방향):")
print("[[1, 2, 3, 4] → [10]")
print(" [5, 6, 7, 8] → [26]")
print(" [9,10,11,12] → [42]")

print("\n" + "=" * 70)
print("4. 다양한 함수에서 axis 사용 예시")
print("=" * 70)

data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

print(f"\n데이터:\n{data}\n")

# 평균 (mean)
print(f"전체 평균: {np.mean(data)}")
print(f"각 열의 평균 (axis=0): {np.mean(data, axis=0)}")
print(f"각 행의 평균 (axis=1): {np.mean(data, axis=1)}")

# 최댓값 (max)
print(f"\n전체 최댓값: {np.max(data)}")
print(f"각 열의 최댓값 (axis=0): {np.max(data, axis=0)}")
print(f"각 행의 최댓값 (axis=1): {np.max(data, axis=1)}")

# 표준편차 (std)
print(f"\n전체 표준편차: {np.std(data):.2f}")
print(f"각 열의 표준편차 (axis=0): {np.std(data, axis=0)}")
print(f"각 행의 표준편차 (axis=1): {np.std(data, axis=1)}")

print("\n" + "=" * 70)
print("5. 3D 배열에서 axis 사용")
print("=" * 70)

# 3D 배열: (2, 3, 4) - 2개 층, 각 층은 3행 4열
cube = np.array([[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]],
                 
                 [[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]]])

print(f"\n3D 배열 shape: {cube.shape}")
print(f"첫 번째 층:\n{cube[0]}")
print(f"\n두 번째 층:\n{cube[1]}")

print(f"\nnp.sum(cube, axis=0).shape: {np.sum(cube, axis=0).shape}")
print("→ 층을 따라 합산 (2개 층 → 하나로)")
print(f"결과:\n{np.sum(cube, axis=0)}")

print(f"\nnp.sum(cube, axis=1).shape: {np.sum(cube, axis=1).shape}")
print("→ 각 층에서 행을 따라 합산 (3행 → 하나로)")
print(f"결과:\n{np.sum(cube, axis=1)}")

print(f"\nnp.sum(cube, axis=2).shape: {np.sum(cube, axis=2).shape}")
print("→ 각 층, 각 행에서 열을 따라 합산 (4열 → 하나로)")
print(f"결과:\n{np.sum(cube, axis=2)}")

print("\n" + "=" * 70)
print("6. 여러 축을 동시에 지정하기")
print("=" * 70)

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"\n원본 배열:\n{matrix}")
print(f"shape: {matrix.shape}")

# 여러 축 동시 지정 (튜플로)
print(f"\nnp.sum(matrix, axis=(0, 1)): {np.sum(matrix, axis=(0, 1))}")
print("→ 모든 축에 대해 합산 (axis=None과 동일)")

# 3D 배열에서 여러 축 지정
print(f"\nnp.sum(cube, axis=(0, 2)).shape: {np.sum(cube, axis=(0, 2)).shape}")
print(f"결과: {np.sum(cube, axis=(0, 2))}")
print("→ axis 0과 2가 사라지고 axis 1만 남음")

print("\n" + "=" * 70)
print("7. keepdims 파라미터와 함께 사용")
print("=" * 70)

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(f"\n원본 배열:\n{arr}")
print(f"shape: {arr.shape}")

result_normal = np.sum(arr, axis=1)
print(f"\nnp.sum(arr, axis=1):")
print(f"결과: {result_normal}")
print(f"shape: {result_normal.shape}")

result_keepdims = np.sum(arr, axis=1, keepdims=True)
print(f"\nnp.sum(arr, axis=1, keepdims=True):")
print(f"결과:\n{result_keepdims}")
print(f"shape: {result_keepdims.shape}")
print("→ 차원을 유지하여 브로드캐스팅에 유용")

print("\n" + "=" * 70)
print("8. 실전 예제: 학생 성적 데이터 분석")
print("=" * 70)

# 학생 3명, 과목 4개 (수학, 영어, 과학, 역사)
scores = np.array([[85, 90, 78, 92],  # 학생 1
                   [88, 85, 91, 87],  # 학생 2
                   [92, 88, 85, 90]]) # 학생 3

subjects = ['수학', '영어', '과학', '역사']
students = ['학생1', '학생2', '학생3']

print("\n성적표:")
print(f"{'':8}", end='')
for subject in subjects:
    print(f"{subject:>6}", end='')
print()

for i, student in enumerate(students):
    print(f"{student:8}", end='')
    for score in scores[i]:
        print(f"{score:6}", end='')
    print()

# 각 과목별 평균 (axis=0)
subject_avg = np.mean(scores, axis=0)
print("\n과목별 평균 점수:")
for subject, avg in zip(subjects, subject_avg):
    print(f"{subject}: {avg:.1f}점")

# 각 학생별 평균 (axis=1)
student_avg = np.mean(scores, axis=1)
print("\n학생별 평균 점수:")
for student, avg in zip(students, student_avg):
    print(f"{student}: {avg:.1f}점")

# 각 과목별 최고 점수 (axis=0)
subject_max = np.max(scores, axis=0)
print("\n과목별 최고 점수:")
for subject, max_score in zip(subjects, subject_max):
    print(f"{subject}: {max_score}점")

# 각 학생별 최고 점수 (axis=1)
student_max = np.max(scores, axis=1)
print("\n학생별 최고 점수:")
for student, max_score in zip(students, student_max):
    print(f"{student}: {max_score}점")

print("\n" + "=" * 70)
print("9. 누적 연산에서의 axis")
print("=" * 70)

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(f"\n원본 배열:\n{arr}")

print(f"\nnp.cumsum(arr, axis=0) (행 방향 누적 합):")
print(np.cumsum(arr, axis=0))
print("→ 각 열에서 위에서 아래로 누적")

print(f"\nnp.cumsum(arr, axis=1) (열 방향 누적 합):")
print(np.cumsum(arr, axis=1))
print("→ 각 행에서 왼쪽에서 오른쪽으로 누적")

print("\n" + "=" * 70)
print("요약 및 기억할 핵심 포인트")
print("=" * 70)

print("""
1. axis는 연산을 수행할 '방향'을 지정합니다
   - axis=0: 행 방향 (↓)
   - axis=1: 열 방향 (→)
   
2. axis를 지정하면 해당 차원이 '축소'됩니다
   - (3, 4) 배열에 axis=0 적용 → (4,) 결과
   - (3, 4) 배열에 axis=1 적용 → (3,) 결과
   
3. axis=None (기본값)은 전체 배열에 대해 연산
   
4. 여러 축을 동시에 지정 가능: axis=(0, 1)
   
5. keepdims=True로 차원 유지 가능
   
6. 직관적 이해: "axis=n이면 n번째 인덱스를 따라 반복"
   - axis=0: arr[i, :] 형태로 접근하며 연산
   - axis=1: arr[:, j] 형태로 접근하며 연산
""")


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

# print("Variance:", np.var(arr))
# print("\n")
# print("Product:", np.prod(arr))
# print("\n")
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