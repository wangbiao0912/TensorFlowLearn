#coding:utf-8import argparseimport datetimeimport multiprocessing as mpimport osimport sysimport timefrom functools import partialfrom imageOptimization import imgTransferfrom async_call import async_callimport cv2import pytesseractimport threadingclass MyThread(threading.Thread):    def __init__(self,func,args=()):        super(MyThread,self).__init__()        self.func = func        self.args = args    def run(self):        self.result = self.func(*self.args)    def get_result(self):        try:            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误        except Exception:            return Nonedef format_time(frame_index, fps):    # hours = second // 3600    # minutes = (second - hours * 3600) // 60    # second = second - hours * 3600 - minutes * 60    # microsecond = second.microseconds // 1000    # t = datetime.time(hour=hours, minute=minutes, second=second,microsecond=microsecond)    # return datetime.time.isoformat(t)    td = datetime.timedelta(seconds=frame_index / fps)    ms = td.microseconds // 1000    m, s = divmod(td.seconds, 60)    h, m = divmod(m, 60)    return '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)# 删除某目录下特定文件def removeFileInDir(sourceDir):    for file in os.listdir(sourceDir):        file = os.path.join(sourceDir, file)  # 必须拼接完整文件名        if os.path.isfile(file) and file.find(".png") > 0:            os.remove(file)            print(file + " remove succeeded")# 清除重复数据def getAllContent(listContent, listTime):    newlistTime = []    newlistContent = []    for i in range(len(listContent)):        print("---------")        if listContent[i - 1] != listContent[i]:            print(listContent[i])            print(listTime[i])            newlistTime.append(listTime[i])            newlistContent.append(listContent[i])            # 记录当前数据    return newlistTime, newlistContentdef saveTxtFile(contentsAtrr, vedioPath):    txtFile = vedioPath + '/subtitle.txt'    if os.path.exists(txtFile):        os.remove(txtFile)    file_handle = open(txtFile, mode='w')    for i in range(len(contentsAtrr[0])):        file_handle.write(contentsAtrr[0][i] + ' \n')        file_handle.write(contentsAtrr[1][i] + ' \n')def readImageToTxt(ex_folder, imageFilePath, tempTime):    config = ('--oem 1 --psm 3')    out = imgTransfer(ex_folder + imageFilePath)    # text = pytesseract.image_to_string(Image.open(ex_folder + imageFilePath), lang="chi_sim+eng", config=config)    text = pytesseract.image_to_string(out, lang="chi_sim+eng", config=config)    # 删除一些异常符号    exclude_char_list = '\ .:|\'\"[]()~@#$%^&*_+-={};<>/¥'    text1 = ''.join([x for x in text if x not in exclude_char_list])    text1 = text1.replace("\n", "  ")    print('文件名' + ex_folder + imageFilePath + '   时间：' + tempTime + '----->' + text1)    # I.close()    return text1# video_filename = "/Users/wangbiao/Downloads/Video/视频内容分析.mp4"def subtitleStart(video_filename):    mp.freeze_support()    # / usr / bin / tesseract    # pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'    ex_folder = os.path.splitext(video_filename)[0]    # 申明没张图片图片的临时数据    # num_processes = mp.cpu_count()    num_processes = 4    parser = argparse.ArgumentParser()    parser.add_argument("--input", default=video_filename, type=str)    parser.add_argument("--output", default=ex_folder, type=str)    args = parser.parse_args()    start_time = time.time()    vid = cv2.VideoCapture(args.input)    if vid.isOpened():        success = True    else:        success = False        print("读取失败!")        return None    frame_jump_unit = vid.get(cv2.CAP_PROP_FRAME_COUNT) // num_processes    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))    fps = vid.get(cv2.CAP_PROP_FPS)    interval = fps    # 视频总帧数    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))    # 图像尺寸    image_size = (int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)))    print("帧频{},视频总帧数{},图像尺寸{}", fps, total_frames, image_size)    vid.release()    try:        threads = []        for i in range(num_processes):            # t= threading.Thread(target=process_video,args=(i,args, frame_jump_unit, fps,ex_folder,            #                width, height, interval, num_processes,), name=process_video.__name__)            t=MyThread(process_video, args=(i,args, frame_jump_unit, fps,                                 ex_folder,                           width, height, interval, num_processes,))            # t = MyThread(process_video, (i,args, frame_jump_unit, fps,            #                      ex_folder,            #                width, height, interval, num_processes,), process_video.__name__)            threads.append(t)            t.start()        # for i in range(num_processes):  # start threads 此处并不会执行线程，而是将任务分发到每个线程，同步线程。等同步完成后再开始执行start方法        #     print("启动第几个县城%{i}"%i)        #     threads[i].start()        for i in range(num_processes):  # jion()方法等待线程完成            threads[i].join()        print(threads[1].get_result())        print('DONE AT:', time.ctime())        #        # print(" 准备调用进程----")        #        # p = mp.Pool(num_processes)        # # p.join()        # tempAttr = p.map_async(partial(process_video, args=args, frame_jump_unit=frame_jump_unit, fps=fps,        #                           ex_folder=ex_folder,        #                    width=width, height=height, interval=interval, num_processes=num_processes), range(num_processes))        # print(" 结束调用进程----")        # p.close()        # p.join()    except OSError or UnboundLocalError:        print("程序报错了。。。。")        # p.close()        # p.join()    tempAttrlistTime = [[] for i in range(num_processes)]    tempAttrlistContent = [[] for i in range(num_processes)]    for i in range(num_processes):        if threads[i].get_result() is not None:            tempAttrlistTime[i] = threads[i].get_result()[1]            tempAttrlistContent[i] = threads[i].get_result()[2]    # for i in range(len(tempAttr)):    #     tempAttrlistTime[i] = tempAttr[i][1]    #     tempAttrlistContent[i] = tempAttr[i][2]    print(        "Method {}: Input:{}, Output:{}, Time taken: {}".format(            sys.argv[0], args.input, args.output, time.time() - start_time        )    )    # 删除生成的临时图片    # removeFileInDir(os.path.splitext(video_filename)[0])    print('*' * 22)    print(tempAttrlistTime)    print(tempAttrlistContent)    tempListTime = []    tempListContent = []    for i in range(num_processes):        if len(tempAttrlistTime[i]) > 1:            for j in range(len(tempAttrlistTime[i])):                tempListTime.append(tempAttrlistTime[i][j])                tempListContent.append(tempAttrlistContent[i][j])        # 数据读取    contentsAtrr = getAllContent(tempListContent, tempListTime)    print("最后的数据是==========")    print(contentsAtrr[0])    print("==========")    print(contentsAtrr[1])    saveTxtFile(contentsAtrr, os.path.splitext(video_filename)[0])# @staticmethod# @async_calldef process_video(group_number, args, frame_jump_unit, fps, ex_folder, width, height, interval, num_processes):    print("现在是第几个循环")    print(group_number)    print('*' * 22)    listTime = []    listContent = []    vid = cv2.VideoCapture(args.input)    pos_frames = int(frame_jump_unit * group_number)    vid.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)    proc_frames = 0    if not os.path.exists(args.output):        os.makedirs(args.output)    if group_number != num_processes - 1:        not_last_seg = True    else:        not_last_seg = False    ret, frame = vid.read()    while ret:        if not_last_seg and proc_frames == frame_jump_unit:            break        curr_frame = proc_frames + pos_frames        if curr_frame % interval == 0:            tempTime = str(format_time(curr_frame, fps))            # out_file = args.output + '/{:0>6d}.png'.format(proc_frames + pos_frames)            out_file = args.output + '/{:d}.png'.format(curr_frame)            # 对图片进行处理            # img = frame[int(height * 0.75):height, :]            # img = img[0:height, :]            # img = img[:, :, 0]            # 需对图像做二值化处理。由于字幕的白色的，像素值为 255，将二值化阈值设为 220            # thresh = 220            # _, resize_frame = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)            # THRESH_BINARY  INTER_AREA            # frame = cv2.resize(frame, (width, int(frame_height * 0.25)), interpolation=cv2.INTER_AREA            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)            cv2.imwrite(out_file, frame)            tempContent = readImageToTxt(ex_folder, "/%s.png" % str(curr_frame), tempTime)            if (tempContent is not None) and (tempContent.strip() != ''):                listTime.append(tempTime)                listContent.append(tempContent)        ret, frame = vid.read()        proc_frames += 1    vid.release()    return group_number, listTime, listContentif __name__ == "__main__":    video_filename = "/Users/wangbiao/Downloads/demo/video/视频内容分析.mp4"    # video_filename='/Users/wangbiao/Downloads/Video/48.ts'    # video_filename='/Users/wangbiao/Downloads/Video/视频内容分析.mp4'    # video_filename = '/Users/wangbiao/Downloads/demo/video/测试1.mp4'    subtitleStart(video_filename)