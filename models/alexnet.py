import torch.nn as nn
from .base_module import BaseModule

class AlexNet(BaseModule):
    def __init__(self , num_classes : int = 1000 , init_weight : bool = False):
        super(AlexNet, self).__init__()
        layers = []
        layers.extend([
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3 , stride=2)
        ])

        self.skeleton_ = nn.Sequential(*layers)

        self.head_ = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6 , 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight , mode='fan_out' , nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias , 0)
            elif isinstance(m , nn.Linear):
                nn.init.normal_(m.weight , 0 , 0.01)
                nn.init.constant_(m.bias , 0)


    def forward(self, x):
        input_ = x

        return result_