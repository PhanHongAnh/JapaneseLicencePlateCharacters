import csv

class Label:
    def __init__(self):
        with open('../sample/kmnist/k49_classmap.csv', 'r') as csvFile:
            reader = csv.DictReader(csvFile)
            index = []
            codepoint = []
            char = []
            for row in reader:
                key = list(row.keys())
                value = list(row.values())
                index.append(int(value[0]))
                codepoint.append(value[1])
                char.append(value[2])
            self.index = index
            self.codepoint = codepoint
            self.char = char
        csvFile.close()

    def search(self, i):
        return self.char[i]
