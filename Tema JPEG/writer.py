import struct

# https://www.youtube.com/watch?v=sb8CQ9knDgI&list=PLpsTn9TA_Q8VMDyOPrDKmSJYt1DLgDZU4&index=2

class Writer:
    def __init__(self):
        self.buffer = bytearray()
        self.accumulator_byte = 0
        self.acc_occupied_bits = 0

    def write_bits(self, value, size):        
        for i in reversed(range(size)):        
            bit = (value >> i) & 1

            self.accumulator_byte = (self.accumulator_byte << 1) | bit # push bit onto acc byte

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
            padding = (1 << (8 - self.acc_occupied_bits)) - 1
            self.accumulator_byte = (self.accumulator_byte << (8 - self.acc_occupied_bits)) | padding

            self.buffer.append(self.accumulator_byte)

            if self.accumulator_byte == 0xFF:
                self.buffer.append(0x00)

            self.accumulator_byte = 0
            self.acc_occupied_bits = 0

def write_marker(buffer, marker):
    buffer += bytes([0xFF, marker])

def write_length(buffer, val):
    buffer += struct.pack(">H", val) # big endian, ushort

def write_dqt(buffer, table, table_info_byte):
    write_marker(buffer, 0xDB)
    # Length: 2 (length) + 1 (table info) + 64 (data)
    write_length(buffer, 67) 
    buffer.append(table_info_byte) # 2nd nibble of info byte is either 0 for Luma or 1 for Chroma
                                # 1st nibble is 0 in both cases - means 1 byte per value in quant table
    for val in table:
        buffer.append(val)

def write_sof0(buffer, width, height):
    write_marker(buffer, 0xC0)
    # Length: 2 (length) + 1 (precision) + 2 (height) + 2 (width) + 
    # 1 (Nr of components - here always 3) + 3 * 3 (YCbCr components, 3 bytes per component: ID, sampling factor, quant table ID)
    write_length(buffer, 17)
    buffer.append(8)       # Precision

    write_length(buffer, height) # Height
    write_length(buffer, width) # Width

    buffer.append(3)       # Number of components
    
    buffer += bytes([
        1, 0x11, 0,  # Y
        2, 0x11, 1,  # Cb
        3, 0x11, 1   # Cr
    ])

def write_dht(buffer, bits, vals, table_class, table_id):
    write_marker(buffer, 0xC4)
    
    # 2 (length) + 1 (table info) + 16 (BITS array) + sum(BITS) (number of symbols, 1 byte per symbol)
    length = 19 + sum(bits[1:]) 
    write_length(buffer, length)

    buffer.append((table_class << 4) | table_id) # 1st nibble represents DC/AC table, 2nd nibble is table ID

    buffer += bytes(bits[1:])
    buffer += bytes(vals)

def write_sos(buffer):
    write_marker(buffer, 0xDA)
    # Length: 2 (length) + 1 (Nr of components - here always 3) + 3 * 2 (YCbCr components, 2 bytes per component: ID, DC/AC table ID) +
    # 1 (Start of Selection - 0 for baseline) + 1 (End of Selection - 63 for baseline) + 1 (Successive Approx - 0 for baseline)
    write_length(buffer, 12)

    buffer.append(3)       # Number of components in scan
    
    buffer += bytes([
        1, 0x00, # Y:  DC 0, AC 0
        2, 0x11, # Cb: DC 1, AC 1
        3, 0x11  # Cr: DC 1, AC 1
    ])
    
    buffer += bytes([0, 63, 0]) 

def save_jpeg(filename, width, height, entropy_data, q_luma, q_chroma, huffman_tables):
    buffer = bytearray()

    # SOI
    write_marker(buffer, 0xD8)

    # APPN - JFIF (APP0)
    write_marker(buffer, 0xE0)
    write_length(buffer, 16)
    buffer += b'JFIF\x00'
    buffer += bytes([1, 1, 0, 0, 1, 0, 1, 0, 0])

    # DQTs
    write_dqt(buffer, q_luma, 0)
    write_dqt(buffer, q_chroma, 1)

    # SOF0 - baseline
    write_sof0(buffer, width, height)

    # DHTs
    for table in huffman_tables:
        write_dht(buffer, *table)

    # SOS - one scan for baseline
    write_sos(buffer)

    # 7. Entropy Data
    buffer += entropy_data

    # 8. EOI
    write_marker(buffer, 0xD9)

    with open(filename, 'wb') as f:
        f.write(buffer)