def bytes_to_int(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def bytes_to_str_with_size(value_bytes: bytes) -> str:
    return value_bytes.decode().rstrip('\00')