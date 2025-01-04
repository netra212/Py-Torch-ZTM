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

# Creating a range and tensors like.


'''
Sometimes you might want a range of numbers, such as 1 to 10 or 0 to 100.
You can use torch.arange(start, end, step) to do so.
Where:
    start = start of range (e.g. 0)
    end = end of range (e.g. 10)
    step = how many steps in between each value (e.g. 1)
Note: In Python, you can use range() to create a range. However in PyTorch,
torch.range() is deprecated and may show an error in the future.
'''
# Use torch.arange(), torch.range() is depricated.
# Create a range of value from 0 to 10.
print("\n\n---------------------------------")
print("------- Range & tensors Like ------")
print("-----------------------------------\n")
zero_to_10_values = torch.arange(start=1, end=10, step=1)
print("Printing the values from zeros to 10 values: \n", zero_to_10_values)

'''
Sometimes you might want one tensor of a certain type with the same shape as
another tensor.For example, a tensor of all zeros with the same shape as a
previous tensor.
To do so you can use torch.zeros_like(input) or torch.ones_like(input) which
return a tensor filled with zeros or ones in the same shape as the input
respectively.
'''

# Can also create a tensors of zeros similar to another tensor.
ten_zeros = torch.zeros_like(input=zero_to_10_values) # will have same shape.
print("Zero Likes: \n", ten_zeros)


#
print("\n\n---------------------------------")
print("------- Tensor datatypes ------")
print("-----------------------------------\n")
'''
Some are specific for CPU and some are better for GPU.
The most common data type are torch.float32 or torch.float.
This is referred to as "32-bit floating point". But there's also 16-bit
floating point (torch.float16 or torch.half) and 64-bit floating point
(torch.float64 or torch.double).
# Note: An integer is a flat round number like 7 whereas a float has a decimal
# 7.0. The reasons -> precision in computing.
The higher the precision value (8, 16, 32), the more detail and hence data
used to express a number. So, lower precision datatypes are generally faster
to compute on but sacrifice some performance on evaluation metrics like
accuracy (faster to compute but less accurate).
'''

# Default datatype for tensors in float32
float_32_tensors = torch.tensor([3.0, 6.0, 9.0],
                                dtype=None,  # default to None, which is torch.float32 or whatever datatype is passed.
                                device=None, # defaults to None, which uses the default tensor type.
                                requires_grad=False # If True, operations performed on the tensors are recorded.
                        )

print("Shape: ", float_32_tensors.shape, "dtype: ", float_32_tensors, "Device Used: ", float_32_tensors.device)

# Most Common issues: datatype and device issues. For example, one of tensors is torch.float32 and the other is torch.float16 (PyTorch often likes tensors to be the same format). Or one of your tensors is on the CPU and the other is on the GPU (PyTorch likes calculations between tensors to be on the same device).

# 
float_16_tensors = torch.tensor([3.0, 6.0, 9.0],
                                dtype=torch.float16) # torch.half would also work.
print("dtype: ", float_16_tensors.dtype)

# Getting information from tensors
print("\n\n-----------------------------------\n")
print("- Getting information from tensors -")
print("-----------------------------------\n")

'''
# We've seen these before but three of the most common attributes you'll want to find out about tensors are.
    * shape - what shape is the tensor? (some operations require specific shape
    rules).
    * dtype - what datatype are the elements within the tensor stored in ?.
    * device - what device is the tensor stored on? (usually GPU or CPU).
'''

# create a tensor. 
some_tensor = torch.rand(3, 4)

# Find out details about it.
print("Displaying some tensor: \n", some_tensor)
print("Datatypes of tensor: ", some_tensor.dtype)
print("shape of tensors: ", some_tensor.shape)
print("Device:  ", some_tensor.device)

# Questions for myself: "what shape are my tensors? what datatype are they and where are they stored? what shape, what datatype, where where where"


# 
print("\n\n-----------------------------------\n")
print("Manipulating tensors (tensor operations)")
print("-----------------------------------\n")
'''
A model learns by investigating those tensors and performing a series of
operations (could be 1,000,000s+) on tensors to create a representation
of the pattern in the input data.
These operations are often a wonderful dance between:
    * Addition.
    * Substraction.
    * Multiplication (element-wise).
    * Division.
    * Matrix multiplication.
Stacking these building blocks in the right way, you can create the most sophisticated of neural networks.
'''
print("-----------------------------------\n")
print("-------- Basic operations --------")
print("-----------------------------------\n")
print("# Create a tensor of values and add a number to it")
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

print("\n# Multiply it by 10")
print(tensor * 10)

# Notice how the tensor values above didn't end up being tensor([110, 120, 130]), this is because the values inside the tensor don't change unless they're reassigned.

# Tensors don't change unless reassigned
print("Tensor remained same: ")
print(tensor)

# Let's subtract a number and this time we'll reassign the tensor variable.
# Subtract and reassign
tensor = tensor - 10
print("\n\n Tensor after the substraction: ", tensor)

# 
print("\nAdd and reassign")
tensor = tensor + 10
print("Adding and reassinging the tensor: ", tensor)

# PyTorch also has a bunch of built-in functions like torch.mul() (short for multiplication) and torch.add() to perform basic operations.

# Can also use torch functions
print("\nImplementation of the inbuilt function of pytorch, torch.multiply: ", torch.multiply(tensor, 10))

# But the Original tensor did not changed still. 
print("\nOriginal tensors: ", tensor, "<-- Original tensor did not changed yet.")

# However, it's more common to use the operator symbols like * instead of torch.mul().
# Element-wise multiplication (each element multiplies its equivalent, index 0 -> 0, 1 -> 1, 2 -> 2)
print("\n\n# Element-wise multiplication")
print(tensor, "*", tensor)
print("\nEquals:", tensor * tensor)

# 
print("\n-----------------------------------")
print("Matrix multiplication (is all you need)")
print("-----------------------------------")
'''
PyTorch implements matrix multiplication functionality in the torch.matmul() 
method.
The main two rules for matrix multiplication to remember are:
1. The inner dimensions must match:
    * (3, 2) @ (3, 2) won't work
    * (2, 3) @ (3, 2) will work
    * (3, 2) @ (2, 3) will work

2. The resulting matrix has the shape of the outer dimensions:
    * (2, 3) @ (3, 2) -> (2, 2)
    * (3, 2) @ (2, 3) -> (3, 3)

    Note:- "@" in Python is the symbol for matrix multiplication.

'''
# Create a tensor and perform element-wise multiplication and matrix 
# multiplication on it.
tensor = torch.tensor([1, 2, 3])
print("Tensor 1 for Multiplication: ", tensor)


'''
# The difference between element-wise multiplication and matrix multiplication 
# is the addition of values.
# For our tensor variable with values [1, 2, 3]

Operation                    Calculation                      Code
Element-wise multiplication	 [1*1, 2*2, 3*3]    = [1, 4, 9]	  tensor * tensor
Matrix multiplication	     [1*1 + 2*2 + 3*3]  = [14]	      tensor.matmul(tensor)
'''
print("# Element-wise matrix multiplication: ", tensor * tensor)
print("# Matrix multiplication: ", torch.matmul(tensor, tensor))

print("# Can also use the '@' symbol for matrix multiplication, though not recommended: ", tensor @ tensor)

# SHAPE ERRORS.
print("----------------------------------------------------------------")
print("One of the most common errors in deep learning (shape errors)")
print("----------------------------------------------------------------")

# # Shapes need to be in the right way.
tensor_A = torch.tensor([[1, 2],
[3, 4], [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
[8, 11], [9, 12]], dtype=torch.float32)

# print(torch.matmul(tensor_A, tensor_B)) # this will throw an error.

'''
We can make matrix multiplication work between tensor_A and tensor_B by making
their inner dimensions match. One of the ways to do this is with a transpose
(switch the dimensions of a given tensor).
We can performs transposes in PyTorch using either:
    * torch.transpose(input, dim0, dim1) - where input is the desired tensor
        to transpose and dim0 and dim1 are the dimensions to be swapped.
    * tensor.T - where tensor is the desired tensor to transpose.
'''
# View tensor_A and tensor_B.T
print(tensor_A)
tensor_B = tensor_B.T

# The operation works when tensor_B is transposed.
# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B)
print(output) 
print(f"\nOutput shape: {output.shape}")
# You can also use torch.mm() which is a short for torch.matmul().
output = torch.mm(tensor_A, tensor_B)
print(output) 
print(f"\nOutput shape: {output.shape}")

# A matrix multiplication like this is also referred to as the dot product of
# two matrices.


'''
# Neural networks are full of matrix multiplications and dot products.

# The torch.nn.Linear() module also known as a feed-forward layer or fully
# connected layer, implements a matrix multiplication between an input x
# and a weights matrix A.
# y = x . A^T + b
where,
    * x -> input to the layer (deep learning is a stack of layers like
    torch.nn.Linear()) and others on top of each other.
    * A -> weights matrix created by the layer, this starts out as random
           numbers thst get adjusted as neural network learns to better
           represent patterns in the data (notice the "T", that's because
           weights matrix get transposed).
           Note:- we might also often see 'W' or another letter like
            'X' used to showcase the weights matrix.
    * b -> bias term used to slightly offset the weights & inputs.
    * y -> output or patterns.

This above is the linear function with formula like y = m.x + b which is used
to draw a straight line.
'''

# Since the linear layer starts with a random weights matrix, let's make it reproducible.
torch.manual_seed(42)

# This uses matrix multiplication.
linear = torch.nn.Linear(in_features=2,   # in_features = matches inner dimension of input.
                        out_features=6    # out_features = describe outer value.
)

x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# Finding the min, max, mean, sum, etc (aggregation)
# Create a tensor.
print("\n-------------------------------------------------")
print("Finding the min, max, mean, sum, etc (aggregation)")
print("-------------------------------------------------")
x = torch.arange(start=0, end=100, step=10)
print("\nInput X: ", x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}")  # won't work without float
# datatype
print(f"Sum: {x.sum()}")

# Note: We may find some methods such as torch.mean() require tensors to be in
# torch.float32 (the most common) or another specific datatype, otherwise the operation will fail.


print("\n-----------------------")
print("Positional min/max")
print("-----------------------")
# Find the index of a tensor where the max or minimum occurs with
# torch.argmax() and torch.argmin() respectively.


# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")


# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")


print("\n-----------------------")
print("Change tensor datatype")
print("-----------------------")


'''
A common issue with deep learning operations is having your tensors in
different datatypes.
If one tensor is in torch.float64 & another is in torch.float32, you might run
into some errors.
But we can change the datatypes of tensors using torch.Tensor.type(dtype=None)
where the dtype parameter is the datatype you'd like to use.
'''


tensor = torch.arange(10., 100., 10.)
print(tensor.shape, tensor.dtype)

# Now we'll create another tensor the same as before but change its datatype
# to torch.float16.

tensor_float16 = tensor.type(torch.float16)
print(tensor_float16.shape, tensor_float16.dtype)

tensor_int8 = tensor.type(torch.int8)
print(tensor_int8.shape, tensor_int8.dtype)

# A torch.Tensor is a multi-dimensional matrix containing elements of a single
# data types.

print("--------------------------------------------------")
print("Reshaping, stacking, squeezing and unsqueezing")
print("--------------------------------------------------")

'''
-----------
# Method:
-----------
* `torch.reshape(input, shape)` -> Reshapes input to shape (if compatible),
can also use `torch.Tensor.reshape()`.

* `Tensor.view(shape)` -> Returns a view of the original tensor in a different
shape but shares the same data as the original tensor.

* `torch.stack(tensors, dim=0)` -> Concatenates a sequence of tensors along a
new dimension (dim), all tensors must be same size.

* `torch.squeeze(input)`	-> Squeezes input to remove all the dimenions with
value 1.

* `torch.unsqueeze(input, dim)` -> Returns input with a dimension value of 1
added at dim.

* `torch.permute(input, dims)` -> Returns a view of the original input with its dimensions permuted (rearranged) to dims.
'''

# Create a tensor
print("\nTensor Created: ")
x = torch.arange(1., 8.)
print(x, x.shape)

# Adds an extra dimension with reshape.
print("Tensor Reshaped: ")
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

# We can also change the view.
# Change view (keeps same data as original but changes view only).
print("Tensor View: ")
z = x.view(1, 7)
print(z, z.shape)

# changing the view of a tensor with torch.view() really only creates a new
# view of the same tensor.
# Changing z changes x.
z[:, 0] = 5
print(z, " <-----> ", x)

# If we wanted to stack our new tensors on top of itself 5 times, we could do
# with torch.stack().
x_stacked = torch.stack([x, x, x, x, x], dim=0)  # try changing dim to dim=1 & see what happens.
print(x_stacked)

# Question: How about removing all single dimensions from a tensor?
# To do so you can use torch.squeeze() --> this as squeezing the tensor to
# only have dimensions over 1.
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# And to do the reverse of torch.squeeze() you can use torch.unsqueeze() to
# add a dimension value of 1 at a specific index.

