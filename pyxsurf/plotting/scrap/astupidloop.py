"""
create a stupid loop that advances only after key is pressed.

This works, at least for a good range of pause time, however it can have some
    problem with flickering and it blocks the execution (e.g. cannot plot on other windows).
"""
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt


def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        fig.wait=1
        fig.canvas.draw()
        #it is important to intercept window closing by q or click on close button.

plt.ion()
fig, ax = plt.subplots()
fig.wait=0
fig.canvas.mpl_connect('key_press_event', press)

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')
ax.set_title('Press a key')
plt.show()

while fig.wait==0:
    plt.pause(0.001) #this is needed.

