import hashlib

A = [-0.2643638551235199, -1.2456191778182983, -1.056803822517395, 3.0146968364715576, -0.645596981048584]
B = [-0.48921066522598267, -1.2112832069396973, -0.9948464632034302, 2.829745292663574, -0.8454509973526001]
C = [-0.2643638551235199, -1.2456191778182983, -1.056803822517395, 3.0146968364715576, -0.645596981048584]
L = [0.10840184986591339, 0.19878055155277252, -0.6348602175712585, 0.5955255627632141, -0.3160969316959381]

fingerA = [int(round(x)) for x in A]
fingerB = [int(round(x)) for x in B]
fingerC = [int(round(x)) for x in C]
fingerL = [int(round(x)) for x in L]
print(fingerA)
print(fingerB)
print(fingerC)
print(fingerL)

keyA = hashlib.sha256(str(fingerA).encode())
keyB = hashlib.sha256(str(fingerB).encode())
keyC = hashlib.sha256(str(fingerC).encode())
keyL = hashlib.sha256(str(fingerL).encode())
print(keyA.hexdigest())
print(keyB.hexdigest())
print(keyC.hexdigest())
print(keyL.hexdigest())
