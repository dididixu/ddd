# coding=utf-8
from selenium import webdriver  #导入模块
import time
import csv  #存储数据
from lxml import etree

option = webdriver.ChromeOptions()   #网址获取
option.add_argument('headless')  #设置浏览器静默
driver = webdriver.Chrome(options=option)
driver.get('http://data.eastmoney.com/zjlx/zs399006.html')
time.sleep(2)

source = driver.page_source
mytree = etree.HTML(source)
tables = mytree.xpath("//table[@class='tab1']")#定位表格，返回列表
for i in range(len(tables)):#循环表格
    onetable = []
    trs = tables[i].xpath('.//tr')#取出所有tr标签
    for tr in trs:
        ui = []
        for td in tr:
            texts = td.xpath(".//text()")#取出所有td标签下的文本
            mm = []
            for text in texts:
                mm.append(text.strip(""))
            ui.append(','.join(mm))
        onetable.append(ui)#整张表格

with open('stock.csv', 'a', newline='') as file:
    csv_file = csv.writer(file)
    for i in onetable:
        csv_file.writerow(i)

time.sleep(2)
driver.close()

