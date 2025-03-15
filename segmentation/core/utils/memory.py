

class FIFOQueue:
    def __init__(self, capacity):
        self._capacity = capacity
        self._data_field = []

    @property
    def capacity(self):
        return self._capacity

    @property
    def data_field(self):
        return self._data_field

    def occupied(self):
        return len(self._data_field)

    def push(self, data):
        self._data_field.append(data)

        while len(self._data_field) > self.capacity:
            self.pop()

    def pop(self):
        if len(self._data_field) > 0:
            del self._data_field[0]

    def full(self):
        return self.capacity == len(self._data_field)

    def reset(self):
        self._data_field = []

