from abc import ABC, abstractmethod

class Layer (ABC):
    
    @abstractmethod
    def foward(self):
        return NotImplementedError
   
    @abstractmethod
    def _create_weights(self):
        return NotImplementedError
    
    @abstractmethod
    def _net(self):
        # Campo de local induzido
        return NotImplementedError
    
    @abstractmethod
    def backforward(self):
        return NotImplementedError
    