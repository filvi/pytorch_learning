# %% 
# Ref link https://www.youtube.com/watch?v=exaWOE8jvy8
# import all the functions defined for debug and printing output
import torch
import numpy as np

def myspacers(mystr, n, sep):

    print("\n" + sep*n)
    print(mystr)
    print(sep*n + "\n")



# %% 
# ==============================================================================
# Empty tensors ----------------------------------------------------------------
# ==============================================================================
mystr = "Empty tensor"
myspacers(mystr, len(mystr), "=")

mystr = "torch.empty(2, 2) -> 2x2 matrix"
myspacers(mystr, len(mystr), "-")
print(torch.empty(2, 2))
mystr = "torch.empty(1, 3, 2) -> 1x3x2 matrix"
myspacers(mystr, len(mystr), "-")
print(torch.empty(1, 3, 2))
mystr = "torch.empty(4, 3, 2) -> 4x3x2 matrix"
myspacers(mystr, len(mystr), "-")
print(torch.empty(4, 3, 2))
# ==============================================================================




# %% 
# ==============================================================================
# 1 and 0 tensors --------------------------------------------------------------
# ==============================================================================
mystr = "Create tensors of ones and zeros"
myspacers(mystr, len(mystr), '=')

mystr = "torch.ones(4, 3, 2) -> 4x3x2 tensor full of ones as numpy"
myspacers(mystr, len(mystr), '-')
print(torch.ones(4, 3, 2))

mystr = "torch.zeros(4, 3, 2) -> 4x3x2 tensor full of zeros as numpy"
myspacers(mystr, len(mystr), '-')
print(torch.zeros(4, 3, 2))
# ==============================================================================




# %% 
# ==============================================================================
# Data types -------------------------------------------------------------------
# ==============================================================================
mystr = "Data types of tensors"
myspacers(mystr, len(mystr), '=')

mystr = "torch.ones(4, 3, 2, dtype=torch.int)"
myspacers(mystr, len(mystr), '-')
print("dtype: ", torch.ones(4, 3, 2, dtype=torch.int).dtype)

mystr = "torch.ones(4, 3, 2, dtype=torch.float)"
myspacers(mystr, len(mystr), '-')
print("dtype: ", torch.ones(4, 3, 2, dtype=torch.float).dtype)

mystr = "torch.ones(4, 3, 2, dtype=torch.double)"
myspacers(mystr, len(mystr), '-')
print("dtype: ", torch.ones(4, 3, 2, dtype=torch.double).dtype)
# ==============================================================================




# %% 
# ==============================================================================
# Size -------------------------------------------------------------------------
# ==============================================================================
mystr = "Getting the size of a tensor"
myspacers(mystr, len(mystr), '=')

mystr = "torch.tensor(4, 3, 2).size()"
myspacers(mystr, len(mystr), '-')
print(torch.ones(4, 3, 2).size())

mystr = "torch.size(1).size()"
myspacers(mystr, len(mystr), '-')
print(torch.ones(1).size())

mystr = "torch.size(4, 3).size()"
myspacers(mystr, len(mystr), '-')
print(torch.ones(4, 3).size())
# ==============================================================================




# %% 
# ==============================================================================
# From list --------------------------------------------------------------------
# ==============================================================================
mystr = "Converting list to tensor"
myspacers(mystr, len(mystr), '=')
mystr = "torch.tensor([0]) -> 1x1 list to 1x1 tensor"
myspacers(mystr, len(mystr), '-')
print(torch.tensor([0]))

mystr = "torch.tensor([[0, 2], [1, 3]]) -> 2x2 list to 2x2 tensor"
myspacers(mystr, len(mystr), '-')
print(torch.tensor([[0, 2], [1, 3]]))

mystr = "torch.tensor([[[0, 2], [1, 3]], [[0, 2], [1, 3]], [[0, 2], [1, 3]], [[0, 2], [1, 3]]]) -> 4x2x2 list to 4x2x2 tensor"
myspacers(mystr, len(mystr), '-')
print(torch.tensor([[[0, 2], [1, 3]], [[0, 2], [1, 3]], [[0, 2], [1, 3]], [[0, 2], [1, 3]]]))
# ==============================================================================




# %% 
# ==============================================================================
# Random -----------------------------------------------------------------------
# ==============================================================================
mystr = "Random tensors of different dimensions"
myspacers(mystr, len(mystr), '=')
mystr = "torch.rand(1) -> random 1x1 tensor (0 < values < 1)"
myspacers(mystr, len(mystr), '-')
x = torch.rand(1)
print(f"{x}")
print(f"{x.size()}")

mystr = "torch.rand(1, 2) -> random 1x2 tensor (0 < values < 1)"
myspacers(mystr, len(mystr), '-')
x = torch.rand(1, 2)
print(f"{x}")
print(f"{x.size()}")

mystr = "torch.rand(1,3, 2) -> random 1x3x2 tensor (0 < values < 1)"
myspacers(mystr, len(mystr), '-')
x = torch.rand(1,3, 2)
print(f"{x}")
print(f"{x.size()}")

# ==============================================================================




# %% 
# ==============================================================================
# Operations -------------------------------------------------------------------
# ==============================================================================
mystr = "Basic operations on tensors"
myspacers(mystr, len(mystr), '=')
a = torch.ones(2, 2)
mystr = f"------ a ------"
myspacers(mystr, len(mystr), "-")
print(a)
b = torch.ones(2, 2)
mystr = f"------ b ------"
myspacers(mystr, len(mystr), "-")
print(b)
# c = a + b
c = torch.add(a, b)
mystr = f"------ c = a + b ------"
myspacers(mystr, len(mystr), "-")
print(c)

d = c * c
d = torch.multiply(c, c)
mystr = f"------ d = c * c ------"
myspacers(mystr, len(mystr), "-")
print(d)

e = d - b
e = torch.subtract(d, b)
mystr = f"------ e = d - b ------"
myspacers(mystr, len(mystr), "-")
print(e)

f = e / (b + 4)
f = torch.divide(e, (b+4))
mystr = f"------ f = c / b(+4 ) ------"
myspacers(mystr, len(mystr), "-")
print(f)

# ==============================================================================




# %% 
# ==============================================================================
# In place Operations ----------------------------------------------------------
# ==============================================================================
#  all operations with trailing underscore modify the variable that is applied on
mystr = "In place operations"
myspacers(mystr, len(mystr), '=')
a = torch.ones(2, 2)
b = torch.ones(2, 2)
mystr = "b before"
myspacers(mystr, len(mystr), '-')
print(b)
b.add_(a)
mystr = "b after b.add_(a)"
myspacers(mystr, len(mystr), '-')
print(b)
# ==============================================================================




# %% 
# ==============================================================================
# Slicing ----------------------------------------------------------

mystr = "Slicing tensors"
myspacers(mystr, len(mystr), '=')
mystr = "all operations with trailing underscore modify the variable that is applied on"
myspacers(mystr, len(mystr), '-')
a = torch.ones(2, 2)

mystr = "a[:, 1]"
myspacers(mystr, len(mystr), '-')
print(a[:, 1])

mystr = "a[:, 2, 1]"
myspacers(mystr, len(mystr), '-')
a = torch.ones(3, 3, 3)
a = a[:, 2, 1] 
print(a)

mystr = "x[1,1].item()"
myspacers(mystr, len(mystr), '-')
x = torch.rand(4, 5)
print(x)
print(x[1,1].item()) 
mystr = "print just one value, care it can only be used with 1x1 tensor"
myspacers(mystr, len(mystr), '-')
# ==============================================================================




# %%

# ==============================================================================
# View -------------------------------------------------------------------------
# ==============================================================================
mystr = "reshape a vector in desidered dimension, the values must match the size v1*v2*v3...."
myspacers(mystr, len(mystr), "=")
x = torch.rand(5, 4) # 5*4 = 20
y = x.view(20) # 20*1 = 20
mystr = "torch.rand(5, 4) -> 5*4 = 20  == x.view(20) -> 20*1 = 20"
myspacers(mystr, len(mystr), "-")
print(f"{y}\n{y.size()}\n\n")

x = torch.rand(5, 4, 2) # 5*4*2 = 40
y = x.view(40) # 40*1*1 = 40
mystr = "torch.rand(5, 4, 2) -> 5*4*2 = 40  == x.view(40) -> 40*1*1 = 40"
myspacers(mystr, len(mystr), "-")
print(f"{y}\n{y.size()}\n\n")

x = torch.rand(5, 4, 2) # 5*4*2 = 40
y = x.view(10, 2, 2) # 10*2*2 = 40
mystr = "torch.rand(5, 4, 2) -> 5*4*2 = 40  == x.view(10, 2, 2) -> 10*2*2 = 40"
myspacers(mystr, len(mystr), "-")
print(f"{y}\n{y.size()}\n\n")

# we can use -1 value for just one "vector" to avoid doing the math
x = torch.rand(5, 4, 2) # 5*4*2 = 40
y = x.view(10, -1, 1) # 10*{-1}*1 = 40 where {-1} is inferred 4 autmatically
mystr = "torch.rand(5, 4, 2) -> 5*4*2 = 40  == x.view(10, -1, 1) -> 10*{-1}*1 = 40 where {-1} is inferred 4 autmatically"
myspacers(mystr, len(mystr), "-")
print(f"{y}\n{y.size()}\n\n")
# ==============================================================================




# %%
# ==============================================================================
# From tensor to numpy  --------------------------------------------------------
# ==============================================================================

mystr = "From torch to numpy"
myspacers(mystr, len(mystr), "=")
mystr = "beware that if we change one we change the other like in list shallow copy if tensor is running on CPU"
myspacers(mystr, len(mystr), "-")


# before tensor to numpy
x = torch.ones(5)
mystr = "Tensor [x] before"
myspacers(mystr, len(mystr), "-")
print(x, type(x))
y = x.numpy()
mystr = "Numpy [y] before"
myspacers(mystr, len(mystr), "-")
print(y, type(y))


mystr = "x.add_(1) -> Now I change the `x` adding 1 in torch way"
myspacers(mystr, len(mystr), "-")


# after tensor to numpy
x = torch.ones(5)
mystr = "Tensor [x] after"
myspacers(mystr, len(mystr), "-")
print(x, type(x))
y = x.numpy()
mystr = "Numpy [y] after"
myspacers(mystr, len(mystr), "-")
print(y, type(y))


# %%
# ==============================================================================
# From numpy to tensor  --------------------------------------------------------
# ==============================================================================

mystr = "From numpy to torch"
myspacers(mystr, len(mystr), "=")


# after numpy to tensor 
mystr = "Tensor [x] after"
myspacers(mystr, len(mystr), "-")
print(x, type(x))
# after numpy to tensor 
mystr = "Numpy [y] after"
myspacers(mystr, len(mystr), "-")
print(y, type(y))

x = np.ones(4)
y = torch.from_numpy(x)
mystr = "Numpy [x] before"
myspacers(mystr, len(mystr), "-")
print(x, type(x))
mystr = "Tensor [y] before"
myspacers(mystr, len(mystr), "-")
print(y, type(y))


mystr = "x += 1 -> Now I change the `x` adding 1 in numpy way"
myspacers(mystr, len(mystr), "=")
x += 1

mystr = "Numpy [x] after"
myspacers(mystr, len(mystr), "-")
print(x, type(x))
mystr = "Tensor [y] after"
myspacers(mystr, len(mystr), "-")
print(y, type(y))
# ==============================================================================

# %%
mystr = "Enable GPU comutation (needs CUDAS to be installed and a Nvidia GPU)"
myspacers(mystr, len(mystr), "=")
mystr = "Assign the device to a tensor in 2 way"
myspacers(mystr, len(mystr), "-")
print("""
if torch.cuda.is_available():
    
    device = torch.device("cuda")       # setting the device to cuda
    x = torch.ones(5, device=device)    # assign the device for the tensor x
    
    y = torch.ones(5)                   # create a new tensor
    y = y.to(device)                    # assign it to a device later
    print(x)
""")
# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device) 
    print(x)
print(torch.cuda.is_available())

# %%
