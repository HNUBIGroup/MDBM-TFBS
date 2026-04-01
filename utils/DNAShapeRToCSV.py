import csv


def DNAShapeRToCSV(train_path, test_path, shapes, seq_len):
    """
        将DNAShapeR工具得到的shape样本，保存至csv格式的文件
    """
    for shape in shapes:
        input_file = open(train_path + '.' + shape)
        path_file='../Datasets/data/Shape/'
        out_file = csv.writer(open(path_file +'Train_' + shape + '.csv', 'w', newline=''))
        """
            write header
        """
        row = []
        for i in range(seq_len):
            row.append(i+1)

        for line in input_file.readlines():
            """
               文件格式:
                >1
                NA,NA,4.96,.......,4.92,NA,NA
            """
            line = line.replace('\n', '')
            if line[0] == '>':
                out_file.writerow(row)
                row = []
            else:
                line = line.split(',')
                for char in line:
                    if char == 'NA':
                        row.append(float(0))
                    else:
                        row.append(float(char))
        out_file.writerow(row)


    for shape in shapes:
        input_file = open(test_path + '.' + shape)
        path_file='../Datasets/data/Shape/'
        out_file = csv.writer(open(path_file +'Test_' + shape + '.csv', 'w', newline=''))
        """
            write header
        """
        row = []
        for i in range(seq_len):
            row.append(i+1)

        for line in input_file.readlines():
            """
               文件格式:
                >1
                NA,NA,4.96,.......,4.92,NA,NA
            """
            line = line.replace('\n', '')
            if line[0] == '>':
                out_file.writerow(row)
                row = []
            else:
                line = line.split(',')
                for char in line:
                    if char == 'NA':
                        row.append(float(0))
                    else:
                        row.append(float(char))
        out_file.writerow(row)

dataset_name = 'wgEncodeAwgTfbsHaibHepg2Fosl2V0416101UniPk'
DNAShapeRToCSV(train_path='../Datasets/{}/Shape/train.fa'.format(dataset_name),
               test_path='../Datasets/{}/Shape/test.fa'.format(dataset_name),
               shapes=['EP', 'HelT', 'MGW', 'ProT', 'Roll'], seq_len=101)

print('Success in getting shapes!')

