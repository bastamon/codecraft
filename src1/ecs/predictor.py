import os
import datetime
from copy import deepcopy
import math
import random
def days(date):
    year=int(date[0:4])
    month=int(date[5:7])
    day=int(date[8:10])
    return year, month, day

def be_days(date0, date1):
    days1 = datetime.date(days(date0)[0], days(date0)[1], days(date0)[2])
    days2 = datetime.date(days(date1)[0], days(date1)[1], days(date1)[2])
    if str(days1) == str(days2):
        return 0
    else:
        return int(str(abs(days2-days1)).split(' ')[0])
def readData(ecs_array, input_array):
    pline1 = input_array[0]
    pspace_1 = pline1.find(' ')
    pspace_2 = pline1.find(' ', pspace_1 + 1)
    pCPU = int(pline1[0:pspace_1])
    pMEM = int(pline1[pspace_1:pspace_2])
    FlavorNum = int(input_array[1])
    flavorL = input_array[2:2 + FlavorNum]
    flavorList = []
    for i in flavorL:
        Space_1 = i.find(' ')
        Space_2 = i.find(' ', Space_1 + 1)
        Space_3 = i.find('\r\n', Space_2 + 1)
        NUM = int(i[6:Space_1])
#        CPU = int(i[Space_1:Space_2])
#        MEM = int(i[Space_2:Space_3])
        flavorList.append(NUM)
    flavorList = sorted(flavorList)
    DimToBeOptimized = input_array[-3].replace('\r\n', '')
    PredictTime_Begin = input_array[-2].replace('\r\n', '')
    PredictTime_End = input_array[-1].replace('\r\n', '')
    bline = ecs_array[0]
    Space_1 = bline.find('\t')
    Space_2 = bline.find('\t', Space_1 + 1)
    historyTime_Begin = bline[Space_2 + 1:].replace('\r\n', '')

    fline = ecs_array[-1]
    Space_3 = fline.find('\t')
    Space_4 = fline.find('\t', Space_3 + 1)
    historyTime_End = fline[Space_4 + 1:].replace('\r\n', '')

    trainData = [[0] for _ in range(FlavorNum)]
    for i in range(FlavorNum):
        for j in range(be_days(historyTime_Begin,historyTime_End)):
            trainData[i].append(0)
    for i in ecs_array:
        space_1 = i.find('\t')
        space_2 = i.find('\t', space_1+1)
        tempFlavor = int(i[space_1+7:space_2])
        tempTime = i[space_2+1:].replace('\r\n', '')
        if tempTime is not None:
            value = be_days(historyTime_Begin, tempTime)
            if tempFlavor in flavorList:
                ii = flavorList.index(tempFlavor)
                trainData[ii][value] += 1
            else:
                None
        else:
            print('Time data error.\n')

    return trainData, pCPU, pMEM, PredictTime_Begin, PredictTime_End, FlavorNum, DimToBeOptimized, flavorL, flavorList

# def read_lines(file_path):
#   if os.path.exists(file_path):
#       array = []
#       with open(file_path, 'r') as lines:
#           for line in lines:
#               if line !='\r\n':
#                   array.append(line)
#       return array
#   else:
#       print 'file not exist: ' + file_path
#       return None
# escDataPath = '/Users/caozhongjian/Desktop/trainData.txt'
# inputFilePath = '/Users/caozhongjian/Desktop/input.txt'
# resultFilePath = '/Users/caozhongjian/Desktop/output.txt'
#
# esc_infor_array = read_lines(escDataPath)
# input_file_array = read_lines(inputFilePath)

def predict_vm(ecs_lines, input_lines):
    result = []

    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    trainData , pcpu, pmem, predict_begintime, predict_endtime, flavorNum, Dimop, flavorL, flavorList= readData(ecs_lines,input_lines)
    # print trainData
    # print shapeSize(trainData)
    finalData = deepcopy(trainData)
    N = be_days(predict_begintime,predict_endtime)
    N1 = len(trainData[0])/N
    N2 = len(trainData[0])%N
    x = [[] for _ in range(flavorNum)]
    y = [[] for _ in range(flavorNum)]
    flavorDs = []
    for i in finalData:
        t = []
        for j in range(N1):
            tt = sum(i[N2+(j*N):(j+1)*N+N2])
            t.append(tt)
        flavorDs.append(t)
    print flavorDs
    flavorDd = []
    beta = random.uniform(0.4,0.95)
    for row in flavorDs:
        mm = int(round(random.uniform(2,N1)))
        v = sum(row[0:mm])/(mm * 1.0)
        t = []
        for j in range(len(row)):
            v = beta*row[j] + (1 - beta) * v
            t.append(v)
        flavorDd.append(t)
    flavorD = []
    for i in flavorDd:
        flavorD.append(round(i[-1]))
    print flavorD


            # x[i].append(finalData[i][j:j+N])
            # y[i].append(finalData[i][j+N])
    # normlizeData = []
    # Meana = []
    # for i in finalData:
    #     n = int(len(i))
    #     dataMean = reduce(lambda x, y: x + y, i) / n
    #     Meana.append(dataMean)
    #     print round(dataMean)
    #     dev = [(math.pow(j - dataMean, 2)) for j in i]
    #     dev = reduce(lambda x, y: x + y, dev) / n
    #     t = []
    #     for ii in i:
    #         tt = abs(ii - dataMean)
    #         t.append(tt)
    #     normlizeData.append(t)

    # for i in range(flavorNum):
    #     for j in range(N, len(normlizeData[0]) - N):
    #         x[i].append(normlizeData[i][j - N:j])
    #         y[i].append(normlizeData[i][j + N])
    # x_test = []
    # for i in normlizeData:
    #     t = []
    #     t.append(i[-N:])
    #     x_test.append(t)
    # yy = [[] for _ in range(flavorNum)]
    # for i in range(flavorNum):
    #     for k in y[i]:
    #         yy[i].append([k])


    stand = []
    for i in flavorL:
        Space_1 = i.find(' ')
        Space_2 = i.find(' ', Space_1 + 1)
        Space_3 = i.find('\r\n', Space_2 + 1)
        # NUM = int(i[6:Space_1])
        CPU = int(i[Space_1:Space_2])
        MEM = int(i[Space_2:Space_3]) / 1024
        stand.append([CPU, MEM])
    stand = sorted(stand)
    # Vir_Request = [sum(multlists(C, A)), sum(multlists(C, B))]
    # main()
    Acpu = []
    Bram = []
    # C = [sum(i) for i in historyData]
    C = int(sum(flavorD))
    CflavorD = []
    for i in range(len(flavorList)):
        Acpu.append(stand[i][0])
        Bram.append(stand[i][1])
        CflavorD.append(int(flavorD[i]))

    Vir_Request, N_host, cpurate, ramrate, cpueffvir, rameffvir = assembly_result(Acpu, Bram, CflavorD, Dimop, pcpu, pmem)
    # output.txt
    strline = str(C) + '\r\n'
    for i in range(len(flavorList)):
        strline += 'flavor' + str(flavorList[i]) + ' ' + str(int(flavorD[i])) + '\r\n'

    # strline += 'predict data\n'
    strline += '\r\n' + str(N_host) + '\r\n'

    if Dimop == 'CPU'or Dimop=='CPU\n':
        for i, x in enumerate(cpueffvir):
            strline += str(i + 1) + ' '
            for j, y in enumerate(x):
                if y != 0:
                    strline += 'flavor' + str(flavorList[j]) + ' ' + str(y) + ' '
            strline += '\r\n'
    elif Dimop == 'MEM'or Dimop=='MEM\n':
        for i, x in enumerate(rameffvir):
            strline += str(i + 1) + ' '
            for j, y in enumerate(x):
                if y != 0:
                    strline += 'flavor' + str(flavorList[j]) + ' ' + str(y) + ' '
            strline += '\r\n'
    return strline.split('\r\n')
    #return strline
##################fundtion define
##################fundtion define

def resborder(N_host, pCPU, pMEM):
    return int(N_host * pCPU * pMEM)


def min_N_host(vir_request, pCPU, pMEM):
    return int(math.ceil(min(vir_request / pCPU, vir_request / pMEM)))


def max_N_host(vir_request, pCPU, pMEM):
    return int(math.ceil(max(vir_request / pCPU, vir_request / pMEM)))


def multlists(list1, list2):
    if len(list1) == len(list2) and list1!=[] and list2!=[] and type(list1[0]) != list and type(list2[0]) != list :
        productlist = []
        for i in range(len(list1)):
            productlist.append(list1[i] * list2[i])
        return productlist
    else:
        return [0,0]



def MultiplePack(n, m, w, v, num, m1):
    optp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    path = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    virs_count = [0 for _ in range(n)]
    #tempdict is descend order index
    bb = sorted(v, reverse=True)
    vtemp=[i for i in v]
    tempdict = [0 for _ in range(len(vtemp))]
    for j, x in enumerate(bb):
        for i in range(len(vtemp)):
            if vtemp[i] == x:
                tempdict[j] = i
                vtemp[i] = None
                break
    print tempdict


    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # if optp[i][j]<m1:
            max_num_i = int(min(math.floor(1.0 * j / w[tempdict[i-1]]), num[tempdict[i-1]]))
            optp[i][j] = optp[i - 1][j]
            # if optp[i][j]<m1:
            for k in range(max_num_i + 1):
                if j > k * w[tempdict[i-1]] and optp[i][j] < optp[i - 1][j - k * w[tempdict[i-1]]] + k * v[tempdict[i-1]] and optp[i - 1][j - k * w[tempdict[i-1]]] + k * v[tempdict[i-1]] <= m1 :
                    optp[i][j] = optp[i - 1][j - k * w[tempdict[i- 1] ]] + k * v[tempdict[i- 1] ]
                    if  optp[i - 1][j - k * w[tempdict[i-1]]]!=0 :
                        path[i][j] = k

        #pkcount[i-1] = optp[i][m] - optp[i - 1][m]
    numtemp= [i for i in num]
    for i in range(n - 1, 0, -1):
        for j in range(m-1, 0, -w[tempdict[i-1]]):
            if path[i][j] != 0 and numtemp[tempdict[ i -1]]:
                numtemp[tempdict[i -1]] -=1
            elif numtemp[tempdict[i -1]]*w[tempdict[i-1] ]< m + numtemp[tempdict[i -1]-1]:
                virs_count[tempdict[i -1]] = numtemp[tempdict[i -1]]#(optp[i][j] - optp[i - 1][j-w[tempdict[i]]]) / v[tempdict[i] ]
            break

    #mc
    # mc=m-1
    # for i in range(n-1, 0, -1):
    #     if mc>0:
    #         for j in range(mc, 0, -1):
    #             if optp[i][mc] > optp[i - 1][j-w[tempdict[i- 1]]] :
    #                 # and optp[i][j] > path[i][j]*optp[i - 1][j-w[tempdict[i- 1]]]:
    #                 virs_count[tempdict[i - 1]] = path[i-1][j]
    #                 #virs_count[tempdict[i - 1]] = optp[i][j]/v[tempdict[i-1]]
    #                 #(optp[i][j] - path[i][j]*optp[i - 1][j-w[tempdict[i- 1]]]) / v[tempdict[i- 1] ]
    #                 mc = j
    #                 break

        # if optp[i][j]>56:
        #     print (optp[i][j])
        # else:
        #     print(optp[i][j])
    max_value = optp[n][m]
    # virs_count = [0 for row in range(len(pkcount)+1)]
    # for i in range(n):
    #     if pkcount[i] != 0:
    #         virs_count[i] = pkcount[i] / v[i]
            # virs_count[i] = pkcount[i] / v[i]


    # for i in range(n+1):
    #     for j in range(m+1):
    #         if max_value<=optp[i][j] and max_value >limits:
    #             max_value=optp[i][j]
    # pkg = [i[m] for i in pkcount]

    return max_value, virs_count


# main#############print ('hello')
def assembly_result(A, B, Cflav ,dimop,pcpu,pmen):
    C =[i for i in Cflav]
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


                if dimop == 'MEM' or dimop == 'MEM\n':
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

            # if costCPU <= i * pCPU:
            #     cpurate.append(costCPU / i / float(pCPU))
            #     cpueffvir.append(cpuvir)
            # if costRAM <= i * pMEM:
            #     ramrate.append(costRAM / i / float(pMEM))
            #     rameffvir.append(ramvir)

    # for i in range(len(cpurate)-1, 0, -1):
    #     for j in range(len(C)):
    #         cpueffvir[i][j] = cpueffvir[i][j] - cpueffvir[i - 1][j]
    #         rameffvir[i][j] = rameffvir[i][j] - rameffvir[i - 1][j]
    # print score

    # score=[]
    # for i in rate:
    #     score.append(f(rate))#f

    print ('CPU' + 'OR RAM ')
    # for i in range(len(cpurate)):
    #     print (str(cpurate[i]) + '\t\t' + str(ramrate[i]))
    print ('CPU  vm  list:')
    print(cpueffvir)
    print ('RAM  vm  list:')
    print(rameffvir)
    #N_host if host-1

    for i in range(len(cpueffvir)-1,0,-1):
        if cpueffvir[-i] == [0 for col in range(len(cpueffvir[0]))]:
            cpueffvir.pop()
    for i in range(len(rameffvir) - 1, 0, -1):
        if rameffvir[-i] == [0 for col in range(len(rameffvir[0]))]:
            rameffvir.pop()
    N_host=max(len(cpueffvir),len(rameffvir))
    return Vir_Request, N_host, cpurate, ramrate, cpueffvir, rameffvir
# f = open(resultFilePath, 'w+')
# strline = predict_vm(esc_infor_array, input_file_array)
# f.writelines(strline)
# #print predict_vm(esc_infor_array, input_file_array)
# f.close()
