#from pool/pipeline 
import re
from itertools import chain
import os
"""extract from a list of notebooks 
a set of variables as in configuration files."""

prefixes=['gfile','cfile','markers1','markers2','gscale','mscale','rect1','c2file']
#notebooks=['OP1S16_analysis.ipynb']
notebooks=[l.strip() for l in open('listdir.dat','r').readlines()]

outconf=[]
for nb in notebooks:
	key=os.path.splitext(os.path.basename(nb))[0]
	l=[ll for ll in open(nb,'r').readlines()]	
	a=[re.findall(r'"(.*?)"', ll.strip()) for ll in l]
	b=list(chain.from_iterable([aa for aa in a if len(aa)>0]))
	b=[ll[:-2] if ll.endswith('\\n') else ll for ll in b]
	#a=[re.search(r'\"(.+?)\"',ll) for ll in l]
	#b=[aa.string[aa.start():aa.end()] for aa in a if aa is not None]
	good=[]
	for prefix in prefixes:
		c=[bb.strip() for bb in b if bb.strip().startswith(prefix)]
		good.extend(c)
	outconf.append("["+key+"]")
	outconf.append("\n".join(good))
	outconf.append("")

outconf='\n'.join(outconf)
open('outconf.ini','w').write(outconf)
print(outconf)
	