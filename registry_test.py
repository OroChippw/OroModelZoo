from models import build_backbone



class TempModel():
    def __init__(self , backbone):
        self.backbone = build_backbone(backbone)

    def run(self):
        print('Backbone : {}'.format(self.backbone))

if __name__ == '__main__':
    temp = TempModel(**cfg)
    temp.run()