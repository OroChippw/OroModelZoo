from Registry import Registry , build_from_cfg

FRUIT = Registry('fruit')
FOOD = Registry('food')

def build(cfg , registry , default=None):
    return build_from_cfg(cfg , registry , default)

def build_fruit(cfg):
    return build(cfg , FRUIT)

def build_food(cfg):
    return build(cfg , FOOD)

# @FOOD.register_module()
# class Rice():
#     def __init__(self , name) -> None:
#         self.name = name
        
# @FRUIT.register_module()
# class Apple():
#     def __init__(self , name) -> None:
#         self.name = name 