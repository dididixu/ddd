from datetime import datetime, date

time_1 = '2020-03-02 15:00:00'
time_2 = '2020-03-02 16:00:00'

time_1_struct = datetime.strptime(time_1, "%Y-%m-%d %H:%M:%S")
time_2_struct = datetime.strptime(time_2, "%Y-%m-%d %H:%M:%S")
seconds = (time_2_struct - time_1_struct).seconds
print(seconds / 3600)
