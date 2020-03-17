from pySurf.points import *
from dataIO.fn_add_subfix import fn_add_subfix

def tomg_file(f):
    """convert matrix data file to 'smart' format martagiorgia.
    See format specs at the end.
    """
    p=get_points(f,delimiter=' ')
    dx,dy=points_find_grid(p,'step')[1]
    pp=matrix_to_points2(points_autoresample(p)[0])
    pp=pp[pp[:,0].argsort(kind='mergesort'),:]*[1,1,1000.]
    save_points(fn_add_subfix(f,'','.mgf'),pp,fill_value='32767',
        header='\n'.join([str(dx),str(dy)]),newline='\n',
        delimiter='\n',comments='')

def tomg(f,p):
    """convert points [Nx3] data to 'smart' format martagiorgia.
    See format specs at the end.
    """
    #p=get_points(f,delimiter=' ')
    dx,dy=points_find_grid(p,'step')[1]
    pp=matrix_to_points2(points_autoresample(p)[0])
    pp=pp[pp[:,0].argsort(kind='mergesort'),:]*[1,1,1000.]
    save_points(fn_add_subfix(f,'','.mgf'),pp,fill_value='32767',
        header='\n'.join([str(dx),str(dy)]),newline='\n',
        delimiter='\n',comments='')
    
if __name__=="__main__":
    file=r'170922_PCO1.2S03_RefSub_4in.dat'
    tomg_file(file)
    
'''
--------------------------------------------------------------------------------------------------
Questo formato e' adatto per passare sia superfici che funzioni di rimozione che matrici dei tempi
--------------------------------------------------------------------------------------------------

1) all'inizio del file mettere due righe contenenti il sampling in x e il sampling in y.

2) formato a 1 colonna, con rispettivamente in sequenza x,y,z. Poi si ripete per ogni punto successivo

3) i valori di x ed y sono interi, uguali agli indici dell'array ed iniziano entrambi da 0 

4) il valore di y varia piu' velocemente di x

5) alla fine di ogni riga mettere un carriage return + un line feed

6) il valore di z e' espresso in:
	- nanometri per le superfici (3 cifre decimali)
	- nanometri/sec per le funzioni di rimozione (tre cifre decimali)
	- secondi per le time matrix (3 cifre decimali)

7) laddove non c'e' informazione usare il valore 32767 come BAD


NOTA: se non si riesce a leggere il file e' possibile che non sia corretto il formato di fine riga che vuole un CRLF.
In tal caso un modo veloce di verificarlo e correggerlo e' tramite il programma Notepad++ che ha comandi specifici al caso citato


----------------------------------------------------------
esempio di file: (sampling di 2.249 in x e 2.249 in y)
----------------------------------------------------------


2.2490
2.2490
0.000	(spiegazione: x)
0.000	(y)
0.000	(z)
0.000	(x)
1.000	(y)
0.000	(z)
0.000	(x)
2.000	(y)
0.000	(z)
0.000	ecc........
3.000
0.000
0.000

'''