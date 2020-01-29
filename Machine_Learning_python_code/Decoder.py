# cbc-implmentation-v1.py
# python 3 required

from array import array
from binascii import hexlify, unhexlify
from base64 import b64encode, b64decode

"""Notes:
The encryption scheme uses
"""


def encrypt(pblks, iv, k):
    iv = 900
    cblks = [iv,]  # Set first value in cbc to iv.

    cblks.append(k^(cblks[0]^pblks[0]))

    for i in range(1, len(pblks)):
        blk = cblks[i] ^ pblks[i]
        cblks.append(k^blk)

    return cblks

def decrypt(ct, key):
    # an array of integers representing the decrypted blocks in eblks
    dblks = []

    #pt=[]
    ct = [900] + ct
    for i in range(1,len(ct)-1):
        dblks.append(key^(ct[i+1]^ct[i]))

    #return pt

    return dblks

def makeblks(m, blksize):
    eblks = []
    bytes_len = blksize//8
    for i in range(0, len(m), bytes_len):
        blk_as_str = m[i:i+bytes_len]
        blk_as_bytes = blk_as_str.encode()
        blk_as_hex = hexlify(blk_as_bytes)
        eblks.append(int(blk_as_hex, 16))

    return eblks

def printblks(blks):
    """Returns a string representation of the given list of integers
    by first converting each int to a hex, stripping off the '0x'
    flag, then converts the hex string to its byte value, then decodes to
    a string.
    """
    m = ''
    try:
        for i in blks:
            hex_str = hex(i)[2:]
            m += unhexlify(hex_str).decode()
    except:
        pass

    return m

def encodeblks(blks, typecode):
    """Builds and returns a base64 encoded string from bytes.
    Returns a base64 encoded string.

    blks: list of blocks (ints)
    typecode: number of bytes to store each integer.

    See: https://docs.python.org/3/library/array.html#module-array
    """

    # Build array object from blks convert to byte array
    arr = array(typecode, blks)
    arrbytes = arr.tobytes()

    # Encode byte array in base64 representation
    arrencoded = b64encode(arrbytes)

    # Return encoded string.
    return arrencoded.decode()

def decodeblks(b64string, typecode):
    """Decodes a base64 encoded string and returns a list of corresponding
    bytes (integers).
    Returns a list of integers.

    b64string: a base64 encoded string
    typecode: number of bytes to store each integer. This must be the same
    value as that used to build array object.
    """

    # Decode base64 string back into bytes.
    arrdecoded = b64decode(b64string)

    # Build a new array object with bytes.
    arr = array(typecode, arrdecoded)

    # Build and return a list of bytes (integers) from array object.
    blks = arr.tolist()
    return blks

if __name__ == '__main__':

    print ('\nDecrypting ...\n')
    estring="hAMAAAAAAAAgInVBAAAAAIQDESkAAAAAdGd9QAAAAADKAxQ3AAAAAH93RBcAAAAAzhIkZAAAAABoeklEAAAAAJhXLCEAAAAAITddcgAAAACQQXwcAAAAACEzXW8AAAAAkEApGwAAAABgJU1vAAAAANhQJRgAAAAAYTlQOAAAAADeS3FLAAAAAGFqBDkAAAAA2AclXwAAAABpdEErAAAAANwHNV8AAAAAYmcUcwAAAADEAzUXAAAAAHZ7RnIAAAAAhgMjHQAAAAAicUNqAAAAAJ9QJg8AAAAAbzlEegAAAADaVSwOAAAAACo6RC4AAAAAnlogXAAAAABuPE81AAAAANVSIVcAAAAAbSQAJAAAAACdTWJNAAAAAG0oAiUAAAAA0AlsSwAAAAB0YR8uAAAAAIQScQ4AAAAAIX8ReAAAAACXETAdAAAAAD5+UD0AAAAAgBY6HQAAAAA+dht5AAAAAIYAOh0AAAAAdmlYdAAAAADDGjwDAAAAAH9uWyMAAAAAyQB6TwAAAAB8aQ9vAAAAAN8HY08AAAAAZnRCOwAAAADDFipfAAAAAGBiRDMAAAAA1A0kEwAAAAB0YUwzAAAAAM0TPlwAAAAAPXdTPgAAAACIADZfAAAAACp0QzEAAAAA2lsxVAAAAAB+el4bAAAAANlbOnMAAAAAbDZUGwAAAADZXyE7AAAAAGwpAEIAAAAAyAhkMAAAAAB1KQBYAAAAAIVccjcAAAAAMCkHQgAAAACVWiYwAAAAADYyRVIAAAAAjkdkOgAAAAA3ZhFbAAAAAIYQMC8AAAAAOWERXAAAAACLCWMvAAAAAC8oB0MAAAAAjVkmLAAAAABzLElFAAAAAMZFHGUAAAAAeTA9HAAAAADIEVhwAAAAAGxlNhIAAAAAzAleMgAAAAB1eyxdAAAAAIUfQT8AAAAAMXslWwAAAACPEwQoAAAAADp6cQgAAAAAmRQdKAAAAAAkfDxcAAAAAIcOUiwAAAAAMmMxRQAAAACMA1xlAAAAADtjKAIAAAAAikIFZwAAAAAtY2AJAAAAAN0HE2wAAAAAYWp7CgAAAADGSx5vAAAAADYiawYAAAAAj0EHZwAAAAAqL28TAAAAAJ9dTmAAAAAAITk7DgAAAADRS19tAAAAAHUrNhkAAAAAxE9aOQAAAAB6bi9XAAAAAMMHWjgAAAAAZiY8VgAAAADEQlk4AAAAAHErLBgAAAAAz19eOAAAAAB6DX8WAAAAAIp+EafDAAAAKRpl9sMAAADZegCXwwAAAGceceTDAAAA2HdQkMMAAAAoBSPlwwAAAJNnQ5fDAAAAYwEs/sMAAACTc0SWwwAAACoTN/TDAAAAjjJFmsMAAAArQ2T1wwAAAJ4uH4/DAAAAOlpxr8MAAACPMwSPwwAAAD5XaK/DAAAAiTgAwcMAAAB5X27hwwAAAMwzAJLDAAAAcltp5sMAAADLNkiBwwAAAG8XLOrDAAAAhWVEgsMAAAA9MGeiwwAAAIhDRsfDAAAANi00psMAAADGSlqGwwAAADYuM/LDAAAAg11ch8MAAAA9My7mwwAAAJlbWMbDAAAAIS15rsMAAADRRBvHwwAAAHNlY6rDAAAAzBcDz8MAAABvfyKhwwAAANkYQoHDAAAAfXoq7cMAAADeW0+IwwAAAGstbufDAAAA1UkEhsMAAAB8JSX1wwAAAM1BVtXDAAAAPS44psMAAACZTlHSwwAAAD0mJ/LDAAAAiFUGmsMAAAA2O3T7wwAAAIsaPNvDAAAALnZPrsMAAACKFm7cwwAAAC94FvzDAAAAmhs3jsMAAAAzbkPvwwAAAMFBQu/DAAAA"

    dblks = decodeblks(estring, 'L')
    for i in range(1,200):
        a = decrypt(dblks,i)
        print(printblks(a))
