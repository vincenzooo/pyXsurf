from Numeric import *
import string

class UserArray:
	def __init__(self, data, typecode=None, copy=1):
		# Needs more testing 
		if typecode != None:
		    data = array(data, typecode, copy=copy)
		else:
		    data = array(data, copy=copy)
		self.__dict__["array"] = data
		self.__dict__["shape"] = self.array.shape
		self.__dict__['name'] = string.split(str(self.__class__))[1]
		if data.typecode() == Complex:
		    self.__dict__['real'] = self._return(self.array.real)
		    self.__dict__['imag'] = self._return(self.array.imag)
		    self.__dict__['imaginary'] = self.imag

	def __setattr__(self,att,value):
	    if att == 'shape':
		self.__dict__['shape']=value
		self.array.shape=value
	    else:
		raise AttributeError, "Attribute cannot be set"

	def __repr__(self):
		return self.name+repr(self.array)[len("array"):]

	def __str__(self):
		return str(self.array)

	def __array__(self,t=None):
		if t: return asarray(self.array,t)
		return asarray(self.array)


	# Array as sequence
	def __len__(self): return len(self.array)

	def __getitem__(self, index): 
		return self._return(self.array[index])

	def __getslice__(self, i, j): 
		return self._return(self.array[i:j])


	def __setitem__(self, index, value): 
	    self.array[index] = asarray(value,self.array.typecode())
	def __setslice__(self, i, j, value): 
	    self.array[i:j] = asarray(value)

	def __abs__(self): return self._return(absolute(self.array))
	def __neg__(self): return self._return(-self.array)

	def __add__(self, other): 
		return self._return(self.array+asarray(other))
	__radd__ = __add__

	def __sub__(self, other): 
		return self._return(self.array-asarray(other))
	def __rsub__(self, other): 
		return self._return(asarray(other)-self.array)

	def __mul__(self, other): 
		return self._return(multiply(self.array,asarray(other)))
	__rmul__ = __mul__

	def __div__(self, other): 
		return self._return(divide(self.array,asarray(other)))
	def __rdiv__(self, other): 
		return self._return(divide(asarray(other),self.array))

	def __mod__(self, other): 
		return self._return(remainder(self.array,asarray(other)))
	def __rmod__(self, other): 
		return self._return(remainder(asarray(other),self.array))

	def __pow__(self,other): 
		return self._return(power(self.array,asarray(other)))
	def __rpow__(self,other): 
		return self._return(power(asarray(other),self.array))

	def __sqrt__(self): 
		return self._return(sqrt(self.array))

	def tostring(self): return self.array.tostring()

	def byteswapped(self): return self._return(self.array.byteswapped())
	def astype(self, typecode): return self._return(self.array.astype(typecode))
   
	def typecode(self): return self.array.typecode()
	def itemsize(self): return self.array.itemsize()
	def iscontiguous(self): return self.array.iscontiguous()

	def _return(self, a):
	    if len(shape(a)) == 0: 
		return a
	    else: 
		r = self.__class__(())
		r.__dict__['array'] = a
		r.__dict__['shape'] = shape(a)
		return r




def is_subclass(a, b):
    """Determine if class(a) is a subclass of class(b)."""
    try: raise a
    except b.__class__: return 1
    except: return 0

def arg_class(value, *args):
    """Chose array or a subclass of UserArray as a return type."""
    import types, Numeric
    # Check if sequence, if not just return it
    try: len(value)
    except: return value
    # Determine return type
    instances = []
    for arg in args:
	if type(arg) == types.InstanceType:
	    instances.append(arg)
    if len(instances) == 0:
	return asarray(value)
    else:
	# Return the class that is lowest in the inheritance hiearchy.
	lowest = instances[0]
	for inst in instances[1:]:
	    if is_subclass(inst, lowest):
		lowest = inst
	    elif not is_subclass(lowest, inst):
		raise ValueError, "Mismatched classes"		
	return lowest.__class__(value)

import Numeric

# Shadow the structural functions.
# (Function, arrays_args, input_args, output_args(no defaults...))
shadow_params = (("take", "(a, indices)", "a, indices, axis=0","a, indices, axis" ),
		 ("reshape", "(a, shape)", "a, shape", "a, shape"),
		 ("resize", "(a, shape)", "a, shape", "a, shape"),
		 ("transpose", "(a,)", "a, axis=None", "a, axis",),
		 ("repeat", "(a, repeats)", "a, repeats, axis=0", "a, repeats, axis"),
		 ("choose", "(a,)+tuple(b)", "a, b", "a, b"),
		 ("concatenate", "tuple(a)", "a, axis=0", "a, axis"),
		 ("diagonal", "(a,)", "a, k=0", "a, k"),
		 ("ravel", "(a,)", "a", "a"),
		 ("nonzero", "(a,)", "a", "a"),
		 ("where", "(con, t, f)", "con, t, f", "con, t, f"),
		 ("compress", "(a, con)", "a, con, axis=0", "a, con, axis"),
		 ("trace", "(a,)", "a, k=0", "a, k"),
		 ("sort", "(a,)", "a, axis=-1", "a, axis"),
		 # not argsort
		 ("searchsorted", "(a, values)", "a, values", "a, values"),
		 # not argmax
		 # not argmin
		 )

for param in shadow_params:
    dict = {"name" : param[0],
	    "doc" : globals()[param[0]].__doc__,
	    "arrays" : param[1],
	    "inargs" : param[2],
	    "outargs" : param[3] }

    exec 'def %(name)s(%(inargs)s):\n' \
	 '    """%(doc)s"""\n' \
	 '    return apply(arg_class,(Numeric.%(name)s(%(outargs)s),)+%(arrays)s)\n'\
	 % dict

class BinaryUserFunc:
    """Wrapper for binary ufuncs"""
    def __init__(self, ufunc):
	self.ufunc = ufunc
	
    def __call__(self, a, b, c=None):
	if c == None:
	    return arg_class(self.ufunc(a,b), a, b)
	elif type(c) == ArrayType:
	    return arg_class(self.ufunc(a,b,c), a, b, c)
	else:
	    c[:] = self.ufunc(a,b)
	    return arg_class(c, a, b, c)

    def reduce(self, a, axis=0):
	return arg_class(self.ufunc.reduce(a, axis), a)

    def accumulate(self, a, axis=0):
	return arg_class(self.ufunc.accumulate(a), a)

    def outer(self, a, b):
	return arg_class(self.ufunc.outer(a, b), a, b)

    def reduceat(self, a, indices, axis=0):
	return arg_class(self.ufunc.reduceat(a, indices, axis), a, indices)

class MonaryUserFunc:
    """Wrapper for monary(?) ufuncs"""
    def __init__(self, ufunc):
	self.ufunc = ufunc
	
    def __call__(self, a, c=None):
	if c == None:
	    return arg_class(self.ufunc(a), a, b)
	elif type(c) == ArrayType:
	    return arg_class(self.ufunc(a,c), a, c)
	else:
	    c[:] = self.ufunc(a)
	    return arg_class(c, a,c)

    def reduce(self, a, axis=0):
	raise ValueError, "reduce only supported for binary functions"

    def accumulate(self, a, axis=0):
	raise ValueError, "accumulate only supported for binary functions"

    def outer(self, a, b):
	raise ValueError, "outer only supported for binary functions"

    def reduceat(self, a, indices, axis=0):
	raise ValueError, "reduceat only supported for binary functions"


for ufunc in ("add", "subtract", "multiply", "divide", "remainder", 
	      "power","maximum", "minimum",):
    exec "%s = BinaryUserFunc(%s)" % ((ufunc,)*2)

for ufunc in ("arccos", "arcsin",  "arctan", "cos", "cosh", "exp", 
	      "log", "log10", "sin", "sinh", "sqrt", "tan", "tanh",  "conjugate"):
    exec "%s = MonaryUserFunc(%s)" % ((ufunc,)*2)

#############################################################
# Test of class UserArray
#############################################################
if __name__ == '__main__':
	import Numeric

	temp=reshape(arange(10000),(100,100))

	ua=UserArray(temp)
	# new object created begin test
	print dir(ua)
	print shape(ua),ua.shape # I have changed Numeric.py

	ua_small=ua[:3,:5]
	print ua_small
	ua_small[0,0]=10  # this did not change ua[0,0], wich is not normal behavior
	print ua_small[0,0],ua[0,0]
	print sin(ua_small)/3.*6.+sqrt(ua_small**2)
	print less(ua_small,103),type(less(ua_small,103))
	print type(ua_small*reshape(arange(15),shape(ua_small)))
	print reshape(ua_small,(5,3))
	print transpose(ua_small)