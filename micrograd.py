import math
import random
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

    #取相反数
    #-a
    def __neg__(self):
        return self*-1
    
    #魔术方法实现运算符重载
    #a+b；a+2
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+')

        def _backward():  
            #这里所运用的语法是python的闭包
            #python将函数视作对象
            #所以可以直接传递函数到下一个对象
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out

    #备用方法
    #2+a
    def __radd__(self,other):
        return self+other

    #a-b
    def __sub__(self,other):
        return self+(-other)
        
    #魔术方法实现运算符重载
    #a*b;a*2
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    #备用方案
    #2*a
    def __rmul__(self,other):
        return self*other

    #除法
    #a/b
    def __truediv__(self,other):
        return self*(other**-1)
    
    #幂次函数（other此刻是int或float）
    #a**2
    def __pow__(self,other):
        assert isinstance(other,(int ,float)),"only support int/float powers for now"
        out=Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad+=other*self.data**(other-1)*out.grad
        out._backward=_backward
        return out
        
    #exp函数，计算e的x次方
    def exp(self):
        out=Value(math.exp(self.data),(self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad
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

class Neuron:#单个神经元
    def __init__(self,nin):#获取神经元的输入数
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]#random.uniform:均匀随机取值
        self.b=Value(random.uniform(-1,1))

    #魔术方法call
    #这样的话可以实现一个Neuron类的实现，如N
    #可以调用例如N(x)的方法来作为函数
    def __call__(self,x):
        act=sum((xi*wi for xi,wi in zip(self.w,x)),self.b)#这里的x可以为Value也可以是int或float
        out=act.tanh()
        return out

class Layer:#单层神经元
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]#获取nout个nin维的神经元

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]#每一层去call一下，意味着这nout个神经元接收的输入是一样的
        return outs

        
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
#o.showvalues()

x=[2.0,3.0]
n1=Layer(2,3)
n2=Layer(3,4)
n3=Layer(4,1)
out=n3(n2(n1(x)))
out[0].backward()
out[0].showvalues()