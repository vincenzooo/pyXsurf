class Sample(object):
    def __init__(self,**kwargs):
        for k, v in list(kwargs.items()):
            setattr(self, k, v)        
