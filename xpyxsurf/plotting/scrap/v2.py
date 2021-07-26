import numpy as np
import matplotlib.pyplot as plt

class Quest():
    def __init__(self, N = 5, bias=0):
        self.bias = bias
        self.N = N
        self.resImage = np.zeros(self.N)
        self.resUser  = np.zeros(self.N)
        self.mapping = {"b" : -1,"r" : 1}

    def start(self):
        self.fig, self.ax = plt.subplots()
        self.i = 0
        self.im = self.ax.imshow(np.zeros((3,3)), norm=plt.Normalize(-1,1),cmap="bwr")
        self.cid = self.fig.canvas.mpl_connect("key_press_event", self.user_input)
        self.next_image()
        plt.show()

    def next_image(self):
        im = self.generate_image()
        self.resImage[self.i] = im.mean()
        self.im.set_data(im)
        self.fig.canvas.draw_idle()

    def user_input(self, event=None):
        if event.key == ' ':
            if self.i < self.N-1:
                self.i += 1
                self.next_image()
            else:
                self.show_result()
        elif event.key in self.mapping.keys():
            self.resUser[self.i] = self.mapping[event.key.lower()]
        else:
            return

    def show_result(self):
        self.ax.clear()
        self.ax.scatter(range(self.N), self.resImage, label="images")
        self.ax.scatter(range(self.N), self.resUser, label="user")
        self.ax.axhline(self.bias, color="k")
        self.ax.legend()
        self.fig.canvas.draw_idle()


    def generate_image(self):
        return np.random.normal(self.bias,0.7, size=(3,3))

class clickable(object):
    def start(self):
        self.fig, self.ax = plt.subplots()
        self.i = 0
        self.im = self.ax.imshow(np.zeros((3,3)), norm=plt.Normalize(-1,1),cmap="bwr")
        self.cid = self.fig.canvas.mpl_connect("key_press_event", self.user_input)
        im = np.random.normal(self.bias,0.7, size=(3,3))
        self.im.set_data(im)
        self.fig.canvas.draw_idle()
        plt.show()    
        
class Qtest():
    def __init__(self, N = 5, bias=0):
        self.bias = bias
        self.N = N

    def start(self):
        self.fig, self.ax = plt.subplots()
        self.i = 0
        self.im = self.ax.imshow(np.zeros((3,3)), norm=plt.Normalize(-1,1),cmap="bwr")
        self.cid = self.fig.canvas.mpl_connect("key_press_event", self.user_input)
        im = np.random.normal(self.bias,0.7, size=(3,3))
        self.im.set_data(im)
        self.fig.canvas.draw_idle()
        plt.show()

    def user_input(self, event=None):
        if event.key == ' ':
            self.show_result()
        else:
            return

    def show_result(self):
        self.ax.clear()
        self.ax.plot([1,1],[2,3])
        self.fig.canvas.draw_idle()
        plt.ion()
        print(plt.isinteractive())
        plt.show()
        plt.draw()
        plt.pause(0.0001)

def test_pausetime(p=False):
    import matplotlib.pyplot as plt
    plt.ioff()
    def make_plot():
        plt.plot([1, 2, 3])
    #    plt.show(block=False)  # The plot does not appear.
    #    plt.draw()             # The plot does not appear.
        if p:
            plt.pause(0.001)          # The plot properly appears.
        print('continue computation')

    print('Do something before plotting.')
    # Now display plot in a window
    make_plot()

    answer = input('Back to main and window visible? ')
    if answer == 'y':
        print('Excellent')
    else:
        print('Nope')

def test_interac():
    from matplotlib import pyplot as plt
    print (plt.isinteractive())
    plt.ioff()
    plt.figure()
    plt.plot([1,2,3],[-3,3,-6])
    print (plt.isinteractive())
    plt.ion()
    plt.plot([1,2,3],[3,-3,6])
    plt.show()
    #plt.ion()