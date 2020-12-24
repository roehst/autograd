from enum import Enum


class Op(Enum):
    SCALAR = 0
    ADD = 1
    MUL = 2
    SUB = 3


class Number:
    def __init__(self, number, op: Op = Op.SCALAR, depends=None):
        self.number = number
        self.op = op
        self.depends = depends if depends is not None else []
        self.grad = None

    def __add__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        return Number(self.number + other.number, Op.ADD, [self, other])

    def __sub__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        return Number(self.number - other.number, Op.SUB, [self, other])

    def __mul__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        return Number(self.number * other.number, Op.MUL, [self, other])

    @property
    def dependencies_count(self):
        return 1 + sum(
            c.dependencies_count for c in self.depends
        )

    def backward(self, grad=None):

        if grad is None:
            self.grad = Number(1.0)
        else:
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

        if self.op == Op.ADD:
            self.depends[0].backward(self.grad)
            self.depends[1].backward(self.grad)
        elif self.op == Op.SUB:
            self.depends[0].backward(self.grad * (-1))
            self.depends[1].backward(self.grad * (-1))
        elif self.op == Op.MUL:
            self.depends[0].backward(self.depends[1] * self.grad)
            self.depends[1].backward(self.depends[0] * self.grad)
