import csv
with open('sample_output.csv','r') as csv_file:
  csv_reader = csv.reader(csv_file)

  with open("cleaned_output.csv",'w') as new_file:
    csv_writer = csv.writer(new_file, delimiter=',')
    for line in csv_reader:
      if (len(line) == 13):
        csv_writer.writerow(line)

  
