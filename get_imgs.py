'''
项目名称：
创建时间：
'''

__Author__ = "Shliang"
__Email__ = "shliang0603@gmail.com"



'''
项目名称：爬取公章数据
创建时间：20200514

百度图片检索地址：
https://image.baidu.com/search/acjson

参考：
https://blog.csdn.net/hujn3016/article/details/78614878  # 参考程序
https://www.cnblogs.com/hum0ro/p/9536033.html  # 遇到错误参考，我没有安装依赖，再运行一次就没有报错了

公司章主要有：公章、财务章、法人章、合同专用章、发票专用章

下载公章数据：
搜索关键词：
公章：检索到的基本上是圆形章
发票专用章:检索到的基本上是椭圆形章


数据转换为灰度图：
circle_red: 前三百个数据保持红色，后面的都转换为灰度图

circle_red: 300
cicle_gray: 223
rectangle_red:53
rectangle_gray:53
fingeprint_red:48
fingeprint_gray:48
other:279


# 印章提取:
https://blog.csdn.net/u011436429/article/details/80453822
https://blog.csdn.net/wsp_1138886114/article/details/82858380


20200519  爬取胸牌数据、胡子数据
Keyword:
胸牌

Keyword:
亚洲人胡子、年轻人胡子
搜索一些亚洲名人：周杰伦胡子 刘德华胡子等   胡渣


20200525  爬取帽子数据
Keyword:
空姐、空姐服装、军人贝雷帽  鸭舌帽女生  鸭舌帽男生


20200703
Keyword：
女士工作布鞋
'''

__Author__ = "Shliang"
__Email__ = "shliang0603@gmail.com"

import requests
import re
import os
import cv2


def getIntPages(keyword, pages):
    params = []
    for i in range(30, 30*pages+30, 30):
        params.append({
            'tn':'resultjson_com',
            'ipn': 'rj',
            'ct':'201326592',
            'is': '',
            'fp': 'result',
            'queryWord': keyword,
            'cl': '2',
            'lm': '-1',
            'ie': 'utf-8',
            'oe': 'utf-8',
            'st': '-1',
            'ic': '0',
            'word': keyword,
            'face': '0',
            'istype': '2',
            'nc': '1',
            'pn': i,
            'rn': '30'
        })
    url = 'https://image.baidu.com/search/acjson'
    urls = []
    for i in params:
        content = requests.get(url, params=i).text
        img_urls = re.findall(r'"thumbURL":"(.*?)"', content)
        urls.append(img_urls)
        #urls.append(requests.get(url,params = i).json().get('data'))开始尝试的json提取方法
        #print("%d times : " % x, img_urls)
    return urls

def fetch_img(path,dataList):
    if not os.path.exists(path):
        os.mkdir(path)

    x = 1
    for list in dataList:
        for i in list:
            print("=====downloading %d/3000=====" % (x + 1))
            ir = requests.get(i)
            open(os.path.join(path, '%08d.jpg' % x), 'wb').write(ir.content)
            x += 1



if __name__ == '__main__':
    badge_path = r'D:\dataset\badge'
    url = 'https://image.baidu.com/search/acjson'
    dataList = getIntPages('123', 5)
    fetch_img(badge_path, dataList)



