import math
def MultiplePack(n, m, w, v, num, m1):
    optp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    pkcount = [0 for _ in range(n)]
    virs_count = [0 for _ in range(n)]

    for i in range(1, n + 1):
        # print (time.ctime())
        for j in range(1, m + 1):
            # if optp[i][j]<m1:
            max_num_i = int(min(math.floor(1.0 * j / w[i - 1]), num[i - 1]))
            optp[i][j] = optp[i - 1][j]
            # if optp[i][j]<m1:
            for k in range(max_num_i + 1):
                if j > k * w[i - 1] and optp[i][j] < optp[i - 1][j - k * w[i - 1]] + k * v[i - 1] and optp[i - 1][j - k * w[i - 1]] + k * v[i - 1] <= m1:
                    optp[i][j] = optp[i - 1][j - k * w[i - 1]] + k * v[i - 1]
                    virs_count[i] = k

        # pkcount[i-1] = optp[i][m] - optp[i - 1][m]

        # if optp[i][j]>56:
        #     print (optp[i][j])
        # else:
        #     print(optp[i][j])

    max_value = optp[n][m]
    # virs_count = [0 for row in range(len(pkcount)+1)]
    # for i in range(n):
    #     if pkcount[i] != 0:
    #         virs_count[i] = pkcount[i] / v[i]

    # for i in range(n+1):
    #     for j in range(m+1):
    #         if max_value<=optp[i][j] and max_value >limits:
    #             max_value=optp[i][j]
    # pkg = [i[m] for i in pkcount]

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