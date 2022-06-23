import xlwt

fr = open("C:/Users/Jarvan/Desktop/err.txt", 'r+')
lines = fr.readlines()
#print(lines)

stu = []
stuId = []
severity = []
kind = []

for line in lines:
    items = line.strip().split(",")
    #print(items)
    stu.append(items[0].strip('C:\\Users\\Jarvan\\Desktop\\code2\\'))
    severity.append(items[1])
    kind.append(items[2])

for sid in stu:
    loc1 = sid.find("_")
    sid = sid[:loc1]
    stuId.append(sid)
    #print(sid)

# print(stuId)
# print(severity)
# print(kind)


workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('My Worksheet')
for i in range(0, len(stuId)):
    worksheet.write(i, 0, stuId[i])
    worksheet.write(i, 1, severity[i])
    worksheet.write(i, 2, kind[i])
workbook.save('C:/Users/Jarvan/Desktop/t.xls')


dict()
sildict = dict()
