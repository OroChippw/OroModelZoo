from .builder import FOOD , FRUIT
@FOOD.register_module
class Rice():
    def __init__(self , name) -> None:
        self.name = name
        
@FRUIT.register_module
class Apple():
    def __init__(self , name) -> None:
        self.name = name 