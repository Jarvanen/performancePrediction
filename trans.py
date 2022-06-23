import xlrd
import xlwt

book = xlrd.open_workbook('C:/Users/Jarvan/Desktop/contest_subwithcode.xls')

sheet1 = book.sheets()[0]
col5_values = sheet1.col_values(4)
col1_values = sheet1.col_values(0)
# print('第5列值', col5_values)

for x in range(0, len(col5_values)):
    f = open("C:/Users/Jarvan/Desktop/code/" + str(round(col1_values[x])) + "_" + str(x) + ".c", "w")
    code = col5_values[x].replace("\\\\0", "\\0").replace("\\\\n", "ZZ").replace(r"\n", "\n").replace(r"\t", "\t").replace("ZZ", "\\n")
    f.write(code)
    f.close()

