from builder import build_fruit , build_food

from lunch import lunch

class COOKER():
    def __init__(self , food , fruit) -> None:
        self.food = build_food(food)
        self.fruit = build_fruit(fruit)
        
    def run(self):
        print('主食吃: {}'.format(self.food.name))
        print('水果吃: {}'.format(self.fruit.name))

cook = COOKER(**lunch)
cook.run()
        