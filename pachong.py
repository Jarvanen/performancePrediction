import requests
from bs4 import BeautifulSoup
def get_data(city_name,tim):
    url="http://www.tianqihoubao.com/lishi/"+city_name+"/month/"+tim+".html"
    re=requests.get(url)
    html = re.content.decode('gbk')#规范编码，避免乱码
    soup = BeautifulSoup(html,'html.parser')
    data=soup.find_all('tr')
    for i in range(1,len(data)):#因为data[0]没有气温数据
        temp=data[i].text.split()
        temp1=temp[3][:-1]
        temp2=temp[5][:-1]
        res=0
        if temp1=='' and temp2=='':#后来爬数据发现有某天的气温不存在
            continue
        elif temp1=='':
            res=int(temp2)
        elif temp2=='':
            res=int(temp1)
        else:
            res=(int(temp1)+int(temp2))/2.0#取平均值
        #print(tim,temp1,temp2)
        fp.writelines(str(res)+'\n')
        print(str(res))
fp=open('data.txt','w',encoding='utf-8')
city_name="shanghai"
for year in range(2015,2020):#时间从2015年到2019年
    for month in range(1,13):#时间从1月到12月
        tim=str(year)
        if month<10:
            tim+=("0"+str(month))
        else:
            tim+=str(month)
        get_data(city_name,tim)#获取数据

fp.close()