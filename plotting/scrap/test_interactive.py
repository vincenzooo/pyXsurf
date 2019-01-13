# a simple plot, does it return to the prompt?

#answer: if interactive is off (plt.isinteractive=False), the plotting command doesn't block execution, but doesn't either display the window.
# a subsequent call to plt.show() displays the window, but also block execution.
#if interactive mode is on, there is no need to call plt.show, as the window is immediately displayed in a non blocking way.
#however if the same operation is performed inside a loop and we want the window to update at each iteration (even if there is a subsequent blocking command

#it remains to clarify the difference between show(), draw(), draw_idle() and the deprecated show(block=False).

from matplotlib import pyplot as plt
plt.ion()
plt.close('all')

plt.clf()
for i in range(3):
    plt.plot([1,2,3],[j**i for j in [2,-2,2]],label=i)
    plt.draw_idle()
    answer = input('advance? ')
plt.legend(loc=0)



#plt.show() # interesting this is what shows the window, but also blocks the execution
#plt.draw()
