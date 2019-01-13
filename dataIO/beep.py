
"""
TODO: check if imports winsound (or similar libraries for other OSs,
returns a beep function with the best available beep (ignore extra parameters)

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
"""

def beep(*args,**kwargs):
    print('\a')
    
    