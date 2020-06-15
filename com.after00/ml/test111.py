import requests
def get_poems(word_list):
    url = 'http://116.62.45.168:81/poems'
    user_info = {'word1': word_list[0],
                 'word2': word_list[1],
                 'word3': word_list[2],
                 'word4': word_list[3]}
    print("等待服务器相应……")
    r = requests.post(url, data=user_info)
    print(r)
    print("生成完成……")
    return r.text
if __name__=="__main__":
    word_list=["国","泰","民","安"]
    poems=get_poems(word_list)
    # print (poems)