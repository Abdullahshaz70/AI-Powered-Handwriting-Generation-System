
import torch.nn as nn
import torch

class StyleEncoder(nn.Module):

    def __init__(self):
        super(StyleEncoder , self).__init__()

        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)


        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 8 * 8, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x




if __name__ == "__main__":
    model = StyleEncoder()
    x = torch.randn(32, 1, 128, 128)
    print(model(x).shape)