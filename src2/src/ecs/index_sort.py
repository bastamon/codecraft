a=[1,65,3,8,1,8,6,3]
def index_sort(aim,TF = False):
    b=sorted(aim ,reverse= TF)
    bindex = [0 for _ in range(len(aim))]
    for i, x in enumerate(aim):
        for j,y in enumerate(b) :
            if x == y :
                bindex[j] = i
                b[j]=None
                break
    return bindex ,sorted(aim)

I, B=index_sort(a)
print a
print B
print I