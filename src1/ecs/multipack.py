import math

def multlists(list1, list2):
    if len(list1) == len(list2) and list1!=[] and list2!=[] and type(list1[0]) != list and type(list2[0]) != list :
        productlist = []
        for i in range(len(list1)):
            productlist.append(list1[i] * list2[i])
        return productlist
    else:
        return [0,0]


def MultiplePack(n, m, w, v, num, m1):
   for i in range(n):
       for j, x in enumerate(num):
            if x*w[j]<=pmen and x*v[j]<pcpu:
                virs_count[i]=x






    return max_value, virs_count



def assembly_result(A, B, C ,dimop,pcpu,pmen):
    Vir_Request = [sum(multlists(C, A)), sum(multlists(C, B))]
    N_host = int(math.ceil(max(multlists(Vir_Request, [1 / float(pcpu), 1 / float(pmen)], ))))
    # w=np.multiply(A, B,).tolist()
    cpurate = []
    cpueffvir = []
    ramrate = []
    rameffvir = []
    costCPU = 0
    costRAM = 0
    cpuvir=[]
    ramvir=[]
    if N_host == 0:
        print ('Dont need phycal machine')
    else:
        N_host = N_host <<1
        for i in range(1, N_host ):

            if C==[0 for _ in range(len(C))]:
                break
            else:

                if dimop=='CPU'or dimop=='CPU\n':
                    if cpuvir!=[]:
                         for k in range(0,len(C)):
                            C[k] -= cpuvir[k]
                    costCPU, cpuvir = MultiplePack(len(C), pmen, B, A, C, pcpu)  # costCPU+56
                    cpueffvir.append(cpuvir)
                    cpurate.append(costCPU / float(pcpu))


                if dimop == 'RAM' or dimop == 'RAM\n':
                    if ramvir!=[]:
                        for k in range(0,len(C)):
                            C[k] -= ramvir[k]
                    costRAM, ramvir = MultiplePack(len(C), pcpu, A, B, C, pmen)  # costRAM+128
                    rameffvir.append(ramvir)
                    ramrate.append(costRAM / float(pmen))

            # print('when host='+str(i))
            # print('server:'+str(i*pCPU)+'and'+str(i*pMEM))
            # print('VM:'+str(costCPU)+'and'+str(costRAM))
            #

            # if costCPU <=  pcpu:
            #     cpurate.append(costCPU / i / float(pcpu))
            #     cpueffvir.append(cpuvir)
            # if costRAM <=  pmen:
            #     ramrate.append(costRAM / i / float(pmen))
            #     rameffvir.append(ramvir)

    # for i in range(len(cpurate)-1, 0, -1):
    #     for j in range(len(C)):
    #         cpueffvir[i][j] = cpueffvir[i][j] - cpueffvir[i - 1][j]
    #         rameffvir[i][j] = rameffvir[i][j] - rameffvir[i - 1][j]
    # print score

    # score=[]
    # for i in rate:
    #     score.append(f(rate))#f

    # print ('CPU OR RAM ')
    # for i in range(len(cpurate)):
    #     print (str(cpurate[i]) + '\t\t' + str(ramrate[i]))
    # print ('CPU  vm  list:')
    # print(cpueffvir)
    # print ('RAM  vm  list:')
    # print(rameffvir)
    #N_host if host-1

    for i in range(len(cpueffvir)-1,0,-1):
        if cpueffvir[-i] == [0 for _ in range(len(cpueffvir[0]))]:
            cpueffvir.pop()
    for i in range(len(rameffvir) - 1, 0, -1):
        if rameffvir[-i] == [0 for _ in range(len(rameffvir[0]))]:
            rameffvir.pop()

    N_host = max (len(cpueffvir),len(rameffvir))

    return Vir_Request, N_host, cpurate, ramrate, cpueffvir, rameffvir