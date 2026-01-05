class Writer:
    def __init__(self):
        self.buffer = bytearray()
        self.accumulator_byte = 0
        self.acc_occupied_bits = 0

    def write_bits(self, value, size):        
        for i in reversed(range(size)):        
            self.accumulator_byte = (self.acc_occupied_bits << 1) | ((value >> i) & 1) # push bit onto acc byte

            self.acc_occupied_bits += 1
            if self.acc_occupied_bits == 8:                
                self.buffer.append(self.accumulator_byte)
                # byte stuffing
                if self.accumulator_byte == 0xFF:
                    self.buffer.append(0x00)
                self.accumulator_byte = 0
                self.acc_occupied_bits = 0

    def flush(self):
        if self.acc_occupied_bits > 0:
            self.accumulator_byte <<= (8 - self.acc_occupied_bits)
            self.buffer.append(self.accumulator_byte)
            self.accumulator_byte = 0
            self.acc_occupied_bits = 0