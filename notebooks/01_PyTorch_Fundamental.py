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
# We can do so using torch.rand() and passing in the size parameter.
print("Create a random tensors of size (3,4)")
random_tensor = torch.rand(size=(3, 4))
print("Random Tensor: \n", random_tensor, "\nData Types of random tensors:", random_tensor.dtype)

'''
The flexibility of torch.rand() is that we can adjust the size to be whatever
we want. For example, say you wanted a random tensor in the common image shape
of [224, 224, 3] ([height, width, color_channels]).
'''

'''
# Create a random tensor of size (224, 224, 3)
'''
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print("Size: ", random_image_size_tensor.shape, " , ","Dimension: ", random_image_size_tensor.ndim)

# Zeros & Ones.
print("\n\n---------------------------------")
print("----------- Zeros & Ones --------")
print("-----------------------------------\n")

'''
Sometimes you'll just want to fill tensors with zeros or ones.
This happens a lot with masking (like masking some of the values in one tensor
with zeros to let a model know not to learn them).
-> Create a tensor full of zeros with torch.zeros().
'''
# Create a tensor of all zeros.
zeros = torch.zeros(size=(3, 4))
print("Zeros Tesnsor: \n", zeros,  "\nData Types: ", zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print("\nOnes Tensors: \n", ones, "\n", "\nData Types: ", ones.dtype)


# 