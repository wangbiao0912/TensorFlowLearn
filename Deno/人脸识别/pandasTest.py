import pandas as pd

# 数据分析
df1=pd.read_csv('/Users/wangbiao/Downloads/my_test_new_account_message_201909.csv')#这个会直接默认读取到这个Excel的第一个表单
data1=df1.head()#默认读取前5行的数据
print("获取到所有的值:\n{0}".format(data1))#格式化输出
print(data1.columns)                         #返回全部列名
print(data1.shape)                           #f返回csv文件形状
print(data1.loc[0:2])                        #打印第1到2行
print(df1['F7128271019B01'].str)

print("----------------")

#加载papa.txt,指定它的分隔符是 \t
df2=pd.read_csv('/Users/wangbiao/Downloads/month9.txt',sep='\t')#这个会直接默认读取到这个Excel的第一个表单
data2=df2.head()#默认读取前5行的数据
print("获取到所有的值:\n{0}".format(data2))#格式化输出
print(data2.columns)
rowNum=df2.shape[0] #不包括表头
colNum=df2.columns.size
print('行数',rowNum)
print('列数',colNum)

# print(df2['account|authtime'].str.split('|',expand=True)[0])

print("华丽的分隔线___++++++++")
print(pd.merge(df2,df1,on=df2['account|authtime'].str.split('|',expand=True)[0],how=df1.head()))