# coding=utf-8

import requests
from bs4 import BeautifulSoup

url = 'http://www.mzitu.com/26685'
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 UBrowser/6.1.2107.204 Safari/537.36'}

html = requests.get(url, headers=header)
soup = BeautifulSoup(html.text, 'html.parser')

# 最大页数在span标签中的第10个
pic_max = soup.find_all('span')[10].text
print("--------====--------")
print("源代码",html.text)
print("----------------完成的代码")
print(pic_max)

# 输出每个图片页面的地址
for i in range(1, int(pic_max) + 1):
    href = url + '/' + str(i)
    print(href)