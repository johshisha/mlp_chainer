import matplotlib.pyplot as plt
from chainer import computational_graph as c


# draw a image of handwriting number
def draw_digit_ae(data, n, row, col, _type, size = 28):
    plt.subplot(row, col, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

def draw_digits(y, t):
    plt.figure(figsize=(15,25))

    num = y.data.shape[0]
    for i in range(int(num/10)):
        for j in range (10):
            img_no = i*10+j
            pos = (2*i)*10+j
            draw_digit_ae(t[img_no].data,  pos+1, int(num/10)*2, 10, "ans")

        for j in range (10):
            img_no = i*10+j
            pos = (2*i+1)*10+j
            draw_digit_ae(y[img_no].data, pos+1, int(num/10)*2, 10, "pred")

    plt.show()


def draw_graph(variable):
    g = c.build_computational_graph(variable)
    with open('resource/g.out', 'w') as o:
        o.write(g.dump())