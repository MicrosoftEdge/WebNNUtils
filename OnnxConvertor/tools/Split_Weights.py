with open('D:\Projects\RapidProjects\RapidChat\models\Weights.bin', 'rb') as infile:
    data = infile.read()
    with open('D:\Projects\RapidProjects\RapidChat\models\Weights1.bin', 'wb') as part1:
        part1.write(data[::2])
    with open('D:\Projects\RapidProjects\RapidChat\models\Weights2.bin', 'wb') as part2:
        part2.write(data[1::2])
