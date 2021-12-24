class Logger:
    def __init__(self):
        self.entries = {}

    # string representation of class
    def __repre__(self):
        return self.__str__()

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return str(self.entries)

    def __getitem__(self, item):
        return self.entries[item]

    def append(self, filepath):
        epoch = len(self.entries)
        line = [epoch, self.entries[epoch]['loss_train'], self.entries[epoch]['loss_train_eval'],
                self.entries[epoch]['loss_val']]
        line = [str(item) for item in line]
        line = ', '.join(line)
        line += '\n'
        f = open(filepath, 'a')
        f.write(line)
        f.close()
