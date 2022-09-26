import numpy as np


class Tensor():
    def __init__(self, value, requires_grad=True):
        self.value = np.array(value, dtype=float)
        self.requires_grad = requires_grad

        if requires_grad:
            self.grad = np.zeros(self.value.shape)
        
    def backwards(self, accumulated_gradient=None):
        if self.requires_grad:
            self.grad += accumulated_gradient

    def __add__(self, b):
        return Add(self, b)

    def __sub__(self, b):
        return Add(self, Neg(b))

    def __mul__(self, b):
        return Mul(self, b)

    def __truediv__(self, b):
        return Mul(self,  b ** Tensor(-1, requires_grad=False)) 

    def __pow__(self, b):
        return Pow(self, b)

    def __matmul__(self, b):
        return Dot(self, b)

    def __gt__(self, b):
        return Tensor(self.value > b.value, requires_grad=False)

class Add(Tensor):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value + b.value
        self.grad = np.zeros(self.value.shape)

    def backwards(self, accumulated_gradient=None):
        self.a.backwards(accumulated_gradient)
        self.b.backwards(accumulated_gradient)

class Neg(Tensor):
    def __init__(self, a):
        self.a = a
        self.value = -a.value
        self.grad = np.zeros(self.value.shape)
    
    def backwards(self, accumulated_gradient=None):
        self.a.backwards(-accumulated_gradient)

class Mul(Tensor):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value * b.value
        self.grad = np.zeros(self.value.shape)

    def backwards(self, accumulated_gradient=None):
        self.a.backwards(self.b.value * accumulated_gradient)
        self.b.backwards(self.a.value * accumulated_gradient)

class Pow(Tensor):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value ** b.value
        self.grad = np.zeros(self.value.shape)

    def backwards(self, accumulated_gradient=None):
        self.a.backwards(self.b.value * self.a.value ** (self.b.value-1) * accumulated_gradient)
        self.b.backwards(np.log(self.a.value) * self.a.value ** self.b.value * accumulated_gradient)

class Dot(Tensor):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value @ b.value
        self.grad = np.zeros(self.value.shape)

    def backwards(self, accumulated_gradient=None):
        self.a.backwards(accumulated_gradient @ self.b.value.T)
        self.b.backwards(self.a.value.T @ accumulated_gradient)        

class Sum(Tensor):
    def __init__(self, a):
        self.a = a
        self.value = sum(a.value)
        self.grad = np.zeros(self.value.shape)

    def backwards(self, accumulated_gradient=None):
        self.a.backwards(np.ones(self.a.value.shape) * accumulated_gradient)