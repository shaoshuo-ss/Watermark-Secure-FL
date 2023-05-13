import tenseal as ts
import numpy as np
import time
from phe import paillier

# Setup TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192 * 2 * 2,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40
np.random.seed(0)
start = time.time()

# for i in range(1000):
v1 = np.random.standard_normal(100000)
v2 = np.random.standard_normal(100000)
# encrypted vectors
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

result = enc_v1 * 3.1 + enc_v2 * 1.2
a = result.decrypt()
end = time.time()
b = v1 * 3.1 + v2 * 1.2
print(end - start)
print(a)
print(b)

# public_key, private_key = paillier.generate_paillier_keypair()
# start = time.time()
# enc_v1 = [public_key.encrypt(i) for i in v1]
# enc_v2 = [public_key.encrypt(i) for i in v2]
#
# result = enc_v1 + enc_v2
# result = [private_key.decrypt(i) for i in result]
# end = time.time()
# print(end - start)
