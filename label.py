import csv

with open('../sample/kmnist/k49_classmap.csv', 'r') as csvFile:
    reader = csv.DictReader(csvFile)
    hira_index = []
    hira_char = []
    for row in reader:
        key = list(row.keys())
        value = list(row.values())
        hira_index.append(int(value[0]))
        hira_char.append(value[2])
csvFile.close()

with open('../sample/kmnist/Kanji-10.csv', 'r') as csvFile:
    reader = csv.DictReader(csvFile)
    kanji_index = []
    kanji_char = []
    for row in reader:
        key = list(row.keys())
        value = list(row.values())
        kanji_index.append(int(value[0]))
        kanji_char.append(value[1])
csvFile.close()

def search_hira(i):
    return hira_char[i]

def search_kanji(i):
    return kanji_char[i]
