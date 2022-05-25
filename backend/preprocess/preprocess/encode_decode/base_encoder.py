from typing import Union

def int_to_bytes(x: int, bytes_size: Union[int, None]=None) -> bytes:
    if not bytes_size:
        bytes_size = (x.bit_length() + 7) // 8
    return x.to_bytes(bytes_size, 'big')

def str_to_bytes_with_size(value: str, size=30) -> bytes:
    value_bytes = value.encode()
    if len(value_bytes) >= size:
        return value_bytes[:size]
    return value_bytes + (b'\x00' * (size - len(value_bytes)))