#coding:utf-8from flask import Flaskfrom flask import requestfrom videoToImages import subtitleStartfrom async_call import async_callapp = Flask(__name__)@async_calldef run(video_filename):    print("a function--------------")    subtitleStart(video_filename)    print("函数调用中。。。。。。")@app.route('/startSubtitleMessage', methods=["post"])def startSubtitleMessage():    video_filename = request.values.get('videoFilePath')    video_filename = video_filename.replace('\\', '/')    print("路径是：%s" %(video_filename))    run(video_filename)    print("调用程序成功")    return "success "if __name__ == "__main__":    print("python运行中。。。。。")    # video_filename = '/Users/wangbiao/Downloads/demo/video/视频内容分析.mp4'    # run(video_filename)    app.run(host="0.0.0.0", debug=True, port=9999)