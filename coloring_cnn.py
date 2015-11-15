from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
import cv2
import numpy as np

class ColoringCNN(FunctionSet):
    def __init__(self, c=3, n1=32, n2=16, f1=7, f2=3, f3=5):
        super(ColoringCNN, self).__init__(
            layer1 = F.Convolution2D( c, n1, f1, stride=1, pad=3),
            layer2 = F.Convolution2D(n1, n2, f2, stride=1, pad=1),
            layer3 = F.Convolution2D(n2, c, f3, stride=1, pad=2),
        )
        self.c  =  c
        self.n1 = n1
        self.n2 = n2
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        h = self.layer3(h)

        return F.mean_squared_error(h, t)

def reshape(img):
    res = np.float32(img.transpose(2,0,1))
    shape = res.shape
    return res.reshape(1, shape[0], shape[1], shape[2])

if __name__=='__main__':
    src_img = cv2.imread('./src.png')
    dst_img = cv2.imread('./dst.png')

    x_data = reshape(src_img)
    y_data = reshape(dst_img)

    c_cnn = ColoringCNN()

    optimizer = optimizers.Adam()
    optimizer.setup(c_cnn)
    optimizer.zero_grads()
    loss = c_cnn.forward(x_data, y_data)
    loss.backward()
    optimizer.update()
