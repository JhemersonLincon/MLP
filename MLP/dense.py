from .layer import Layer
from random import uniform

class Dense(Layer):
    
    weights:list = None
    units: int
    activation:str
    name:str
    
    def __init__(self, units = 1, name = 'Dense'):
        self.units = units
        self.name = name
        
    def foward(self, x:list) -> list:
        if self.weights is None: 
            self.weights = [self._create_weights(len(x)) for _ in range(self.units)]
        output = []
        for i in range(self.units):
            net = self._net(x, self.weights[i])
            output.append(net)
        return output
   
    def _net(self, x:list, w:list) -> float:
        x = [1] + x
        print(f"x :     {x}")
        print(f"Weight: {w}")
        net = sum([a * b for a, b in zip(x, w)])
        return self._activation(net)

    def _create_weights(self, inputs) -> list[float]:
        weight = [uniform(-1, 1) for _ in range(inputs+1)]
        return weight
    
    def backforward(self):
        return super().backforward()
    
    def _activation(self, net: float) -> float:
        return max(0, net)  # ReLU
