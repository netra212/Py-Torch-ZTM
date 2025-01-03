import torch
print("-----------------------------------")
print("- Version of torch: -")
print("-----------------------------------")
print(torch.__version__)

# Tensors --> Numerical representation of the data.
# We could represent the images as a tensors. [3, 224, 224] -->
# [colour_channels, height, width]
# Tensors would have three dimensions --> [colour_chhanels, height, width ]

# scalar.
print("-----------------------------------")
print("- Scalar: Contains zero dimensions -")
print("-----------------------------------")
scalar = torch.tensor(7)
print(scalar)

# retrieving the number from the tensor.
print("Items from the scalar: ", scalar.item())

# Vector: Contains single dimension.
print("\n\n-----------------------------------")
print("Vector: Contains single dimension.")
print("-----------------------------------")
vector = torch.tensor([5, 6, 7])
print("\nVector: A single dimension tensor but can contains many number: ", vector)

# Check the number of dimension of the vectors.
print("Dimension of the vector: ", vector.ndim)

# Check the shape of the vector.
print("Shape of the vector. ", vector.shape)

# Matrix.
print("\n\n-----------------------------------")
print("---------------- Matrix ----------")
print("-----------------------------------")
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print("MATRIX: Matrices are as flexible as vectors, except they've got an \n"
      "extra dimensions: ", MATRIX.ndim)

# Check the number of dimensions.
print("Number of Dimensions: ",MATRIX.ndim)

# What shape do you think it will have ?
print("Shape of the Matrix: ",MATRIX.shape)

# We get the output torch.Size([2, 2]) because MATRIX is two elements
# deep and two elements wide.

print("\n\n-----------------------------------")
print("------------- Tensor ----------")
print("-----------------------------------\n")
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print("Tensor Elements: ", TENSOR)
print("Check the Number of dimensions for TENSOR: ", TENSOR.ndim)
print("Shape of the tensor:", TENSOR.shape)
# Output: Shape of the tensor: torch.Size([1, 3, 3])
# 1 -> represent the 1st bracket.
# 3 -> represent the 1 dimension.
# 3 -> represent the 2nd dimension.

'''
As I have used the lowercase letter for the `scalar` & `vector` and uppercase for the `MATRIX` & `TENSOR` because this was on purpose. Also, name matrix and tensors are interchangably which is common in deep learning. Since in PyTorch you're often dealing with torch.Tensors (hence the tensor name), however, the shape and dimensions of what's inside will dictate what it actually is.
'''

# Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers..

# Random Number.
print("\n\n-----------------------------------")
print("------------- Random Number ----------")
print("-----------------------------------\n")

