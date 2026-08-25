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

    #三层parameters函数用于获取一个大的参数列表
    def parameters(self):#第一层
        return self.w+[self.b]

class Layer:#单层神经元
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]#获取nout个nin维的神经元

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]#每一层去call一下，意味着这nout个神经元接收的输入是一样的
        return outs[0] if len(outs)==1 else outs#当最后只有一个结果的时候只返回单独值而不是列表

    def parameters(self):#第二层parameters
        return[p for neurons in self.neurons for p in neurons.parameters()]
        
class MLP:
    def __init__(self,nin,nouts):
        sz=[nin]+nouts#这里是size的意思
        #nin是第一层x的数量
        #转换成列表
        #nouts是每一层神经元的数量
        #组合就是[x,n1,n2,...]
        self.layers=[Layer(sz[i],sz[i+1])for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)#递归一层一层联系上
        return x
    #最终的结果就是输入一个x和结构就可以实现这个MLP
    #最终结果就是最后一层的输出

    def parameters(self):#第三层parameters
        return[p for layers in self.layers for p in layers.parameters()]
        
# x1=Value(2.0,label='x1')
# x2=Value(0.0,label='x2')
# w1=Value(-3.0,label='w1')
# w2=Value(1.0,label='w2')
# b=Value(6.8813735870195432,label='b')
# x1w1=x1*w1
# x1w1.label='x1*w1'
# x2w2=x2*w2
# x2w2.label='x2*w2'
# x1w1x2w2=x1w1+x2w2
# x1w1x2w2.label='x1*w1+x2*w2'
# n=x1w1x2w2+b
# n.label='n'
# o=n.tanh()
# o.label='o'
# o.grad=1.0
# o.backward()
# o.showvalues()

# x=[2.0,3.0]
# n1=Layer(2,3)
# n2=Layer(3,4)
# n3=Layer(4,1)
# out=n3(n2(n1(x)))
# out[0].backward()
# out[0].showvalues()

# x=[2.0,3.0,-4.0]
# nouts=[4,4,1]
# m=MLP(3,nouts)#一个3*4*4*1的神经结构
# out=m[x]
# out.backward()
# outß.showvalues()

#这里xs是四组输入，ys是四组答案，现在的想法是训练一个神经网络，让他接收到xs的输入之后可以去预测对应的ys
#就像llm用大量的输入输出组合去训练模型
print("示例模型=============================================")
xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,1.0]
]
ys=[1.0,-1.0,-1.0,1.0]
n=MLP(3,[4,4,1])#和前面的模型的结构是一样的
ypred=[n(x) for x in xs]
params=n.parameters()
print("xs:",xs)
print("ys:",ys)
print("ypred:",ypred)
h=0.0001
print("300次训练：=============================================")
for i in range(300):#训练300轮
    loss = sum((yout-ygt)**2 for ygt,yout in zip(ys,ypred))
    loss.backward()
    for param in params:
        param.data-=param.grad*h
    ypred=[n(x) for x in xs]
    print(loss)
print("训练结束================================================")
print("ys:",ys)
print("ypred:",ypred)
print("目标化训练：=============================================")
while(loss.data>0.01):#训练到loss<0.01
    loss = sum((yout-ygt)**2 for ygt,yout in zip(ys,ypred))
    loss.backward()
    for param in params:
        param.data-=param.grad*h
    ypred=[n(x) for x in xs]
    print(loss)
print("训练结束================================================")
print("ys:",ys)
print("ypred:",ypred)
print("训练成功")

print("除法模型=============================================")
xs=[]
i=1.0
while(i<10.5):
    i+=1.0
    j=1.0
    while(j<i):
        xs.append([j,i])
        j+=1

ys=[(x[0]/x[1]) for x in xs]
n=MLP(2,[10,10,10,1])
ypred=[n(x) for x in xs]
print("xs:",xs)
print("ys:",ys)
print("ypred:",ypred)
h=0.01
while(loss.data>0.05):
    #forward
    ypred=[n(x) for x in xs]
    loss = sum((yout-ygt)**2 for ygt,yout in zip(ys,ypred))

    #backward
    loss.backward()

    #update
    for param in n.parameters():
        param.data-=param.grad*h
    ypred=[n(x) for x in xs]

    #打印loss
    print(i,loss.data)
print("ys:",ys)
print("ypred:",ypred)
