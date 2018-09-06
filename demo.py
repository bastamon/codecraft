# -*- coding:utf-8 -*-


def getdata(content):
    pi = []
    ai = []
    qi = []
    bi = []
    lines = content.split("\n")
    num = int(lines[0].strip())
    for i, x in enumerate(lines[1:]):
        arry = x.split(" ")
        pi.append(int(arry[0]))
        ai.append(int(arry[1]))
        qi.append(int(arry[2]))
        bi.append(int(arry[3]))
    print(max(bag(num, 120, pi, ai), bag(num, 120, qi, bi)))
    return max(bag(num, 120, pi, ai), bag(num, 120, qi, bi))


def bag(n, c, w, v):
    res = [[-1 for j in range(c + 1)] for i in range(n + 1)]
    for j in range(c + 1):
        res[0][j] = 0
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            res[i][j] = res[i - 1][j]
            if j >= w[i - 1] and res[i][j] < res[i - 1][j - w[i - 1]] + v[i - 1]:
                res[i][j] = res[i - 1][j - w[i - 1]] + v[i - 1]
    return res[n][len(res[n]) - 1]


def main():
    with open("text.txt", 'r') as f:
        content = f.read()
    getdata(content)


if __name__ == '__main__':
    main()
