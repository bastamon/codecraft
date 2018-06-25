#-*- coding=utf-8 -*-



def zhaoqian(money):
    loop = True
    tmp = ['总金额：' + str(money) + '元']

    # 面值列表 单位：元
    cate = (
        100,
        50,
        20,
        10,
        5,
        1,
        0.5,
        0.1
    )

    sy = int(money * 10)
    while loop:
        if sy == 0:
            loop = False
        else:
            for row in cate:
                tmpStr = ''
                jine = int(row * 10)
                if jine >= 10:
                    tmpUn = '元'
                else:
                    tmpUn = '角'

                if sy >= jine and tmpStr == '':
                    m = sy // jine
                    sy = sy % jine
                    if jine >= 10:
                        tmpStr = str(jine // 10) + tmpUn + str(m) + '张'
                    else:
                        tmpStr = str(jine) + tmpUn + str(m) + '张'
                    tmp.append(tmpStr)

    return tmp


a = zhaoqian(88.7)
for x in a:
    print x