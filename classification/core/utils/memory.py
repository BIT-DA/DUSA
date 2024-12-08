

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

        self.pop_first()

        # while len(self._data_field) > self.capacity:
        #     self.pop()

    def pop_first(self):
        self._data_field = self._data_field[-self.capacity:]

    def pop(self):
        if len(self._data_field) > 0:
            self._data_field = self._data_field[1:]

    def full(self):
        return self.capacity == len(self._data_field)

    def reset(self):
        self._data_field = []

