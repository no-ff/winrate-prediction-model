import csv

with open('sample_output.csv','r') as csv_file:
  csv_reader = csv.reader(csv_file)

  with open("cleaned_output.csv",'w',newline='') as new_file:
    csv_writer = csv.writer(new_file, delimiter=',')
    line = ['T1C1','T1C2','T1C3','T1C4','T1C5','T2C1','T2C2','T2C3','T2C4','T2C5','GD','Target']
    csv_writer.writerow(line)
    for line in csv_reader:
      if (len(line) == 13):
        csv_writer.writerow(line[:-2])
