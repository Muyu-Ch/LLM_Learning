import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data  #数值
        self.grad = 0.0  #梯度
        self._prev=set(_children)  #树结构引用
        self._op=_op  #符号
        self.label=label  #标签
        self._backward = lambda: None

    #魔术方法，可以更直观的展示Value
    def __repr__(self):
        return f"Value(data={self.data})"

    #魔术方法实现运算符重载
    def __add__(self,other):
        out = Value(self.data+other.data,(self,other),'+')

        def _backward():  
            #这里所运用的语法是python的闭包
            #python将函数视作对象
            #所以可以直接传递函数到下一个对象
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out

    #魔术方法实现运算符重载
    def __mul__(self,other):
        out = Value(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    #压缩函数
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')

        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        
        return out

    #反向传播函数
    #利用拓扑结构反向传播求导
    def backward(self):
        topo=[]
        visited=set()
        #拓扑函数，将整个拓扑结构存储到列表topo里
        #采用递归的方式对根节点进行调用
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        #调用根节点
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()

    def showvalues(self):
        for child in self._prev:
            child.showvalues()
        print(f"label={self.label},data={self.data},grad={round(self.grad,2)}")
        
x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')
b=Value(6.8813735870195432,label='b')
x1w1=x1*w1
x1w1.label='x1*w1'
x2w2=x2*w2
x2w2.label='x2*w2'
x1w1x2w2=x1w1+x2w2
x1w1x2w2.label='x1*w1+x2*w2'
n=x1w1x2w2+b
n.label='n'
o=n.tanh()
o.label='o'
o.grad=1.0

o.backward()

o.showvalues()