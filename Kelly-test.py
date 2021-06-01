# coding: utf-8
import math
import random
import matplotlib.pyplot as plt


# kelly 公式，b代表赔率，p代表获胜概率
def kelly(b, p):
    return (p * (b + 1) - 1) / b


if __name__ == '__main__':
    # b 赔率，p 获胜概率 n 下注次数
    # 赔率计算方式：赚的钱(包含本金) / 亏损的钱
    # pre_money 本金
    # result 投注比例
    b = 2.0
    p = 0.5
    n = 1000
    result = kelly(b, p)
    pre_money = 10
    count = 0
    print("本金%.4f\n赔率为%.4f\n胜率为%.4f\n投注比例 %.4f" % (pre_money, b, p, result))
    money = pre_money
    x = [1]
    y = [pre_money]
    for i in range(2, n + 1):
        te = random.random()
        if te <= p:
            money *= (1 + result * (b - 1.0))
            # money = money + result * money * (b - 1)
        else:
            money *= (1 - result)
        # x.append(i)
        # y.append(money)
        if n <= 2000:
            print("概率为%.2f" % te, "猜错" if te > 0.5 else "猜对", "剩余金钱：%.2f" % money)
    print("运行%d次后，最后剩余%.2f" % (n, money))

    # plt.plot(x, y)
    # plt.show()

    # if money > pre_money:
    #     count += 1
    # else:
    #     print("最后剩余%.2f" % money)
    # print(count)
