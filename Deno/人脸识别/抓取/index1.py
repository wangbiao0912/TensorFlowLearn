# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import os

all_url = 'https://www.mzitu.com'

# http请求头
Hostreferer = {
    'User-Agent': 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)',
    'Referer': 'http://www.mzitu.com'
}
# 此请求头Referer破解盗图链接
Picreferer = {
    'User-Agent': 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)',
    'Referer': 'http://i.meizitu.net'
}

# 对mzitu主页all_url发起请求，将返回的HTML数据保存，便于解析
start_html = requests.get(all_url, headers=Hostreferer)

# Linux保存地址
# path = '/home/Nick/Desktop/mzitu/'

# Windows保存地址
path = '/Users/wangbiao/Downloads/tempData/'

# 获取最大页数
soup = BeautifulSoup(start_html.text, "html.parser")
page = soup.find_all('a', class_='page-numbers')
max_page = page[-2].text

# same_url = 'http://www.mzitu.com/page/'   # 主页默认最新图片
# 获取每一类MM的网址
same_url = 'https://www.mzitu.com/mm/page/'  # 也可以指定《qingchun MM系列》

for n in range(1, int(max_page) + 1):
    # 拼接当前类MM的所有url
    ul = same_url + str(n)

    # 分别对当前类每一页第一层url发起请求
    start_html = requests.get(ul, headers=Hostreferer)

    # 提取所有MM的标题
    soup = BeautifulSoup(start_html.text, "html.parser")
    all_a = soup.find('div', class_='postlist').find_all('a', target='_blank')

    # 遍历所有MM的标题
    for a in all_a:
        # 提取标题文本，作为文件夹名称
        title = a.get_text()
        if (title != ''):
            print("准备扒取：" + title)

            # windows不能创建带？的目录，添加判断逻辑
            if (os.path.exists(path + title.strip().replace('?', ''))):
                # print('目录已存在')
                flag = 1
            else:
                os.makedirs(path + title.strip().replace('?', ''))
                flag = 0
            # 切换到上一步创建的目录
            os.chdir(path + title.strip().replace('?', ''))

            # 提取第一层每一个MM的url，并发起请求
            href = a['href']
            html = requests.get(href, headers=Hostreferer)
            mess = BeautifulSoup(html.text, "html.parser")

            # 获取第二层最大页数
            pic_max = mess.find_all('span')
            pic_max = pic_max[9].text
            if (flag == 1 and len(os.listdir(path + title.strip().replace('?', ''))) >= int(pic_max)):
                print('已经保存完毕，跳过')
                continue

            # 遍历第二层每张图片的url
            for num in range(1, int(pic_max) + 1):
                # 拼接每张图片的url
                pic = href + '/' + str(num)

                # 发起请求
                html = requests.get(pic, headers=Hostreferer)
                mess = BeautifulSoup(html.text, "html.parser")
                pic_url = mess.find('img', alt=title)
                print(pic_url['src'])
                html = requests.get(pic_url['src'], headers=Picreferer)

                # 提取图片名字
                file_name = pic_url['src'].split(r'/')[-1]

                # 保存图片
                f = open(file_name, 'wb')
                f.write(html.content)
                f.close()
            print('完成')
    print('第', n, '页完成')

#
# import requests
#
# url = 'http://www.mzitu.com'
#
# # 设置headers，网站会根据这个判断你的浏览器及操作系统，很多网站没有此信息将拒绝你访问
# header = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 UBrowser/6.1.2107.204 Safari/537.36'}
#
# # 用get方法打开url并发送headers
# html = requests.get(url, headers=header)
#
# # 打印结果 .text是打印出文本信息即源码
# print(html.text)