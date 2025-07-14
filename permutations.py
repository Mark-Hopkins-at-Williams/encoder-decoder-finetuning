import json
import random
from typing import Dict

def create_random_permutation_with_fixed_points(vocab_size, fixed_points):
    p_domain = sorted(set(range(vocab_size)) - set(fixed_points))
    p_range = [t for t in p_domain]
    random.shuffle(p_range)
    return Permutation(p_domain, p_range)

class Permutation:
    def __init__(self, domain, rng):
        self.domain = domain
        self.range = rng
        self.permutation = dict(zip(domain, rng))
        self.inverse = dict(zip(rng, domain))
    
    def __call__(self, i):
        return self.permutation.get(i, i)
    
    def get_inverse(self):
        return Permutation(self.range, self.domain)
    

def save_permutation_map(pmap : Dict[str, Permutation], filename : str):
    to_serialize = dict()
    for key in pmap:
        info = {'domain': pmap[key].domain, 'range': pmap[key].range}
        to_serialize['|||'.join(key)] = info
    with open(filename, "w") as writer:
        json.dump(to_serialize, writer)
        

def load_permutation_map(filename : str):
    with open(filename, "r") as reader:
        saved_drs = json.load(reader)
    pmap = dict()
    for key in saved_drs:
        dr = saved_drs[key]
        p = Permutation(dr['domain'], dr['range'])
        new_key = tuple(key.split("|||"))
        pmap[new_key] = p
    return pmap
    

if __name__ == "__main__":
    p = create_random_permutation_with_fixed_points(8, [0, 1, 2, 7])
    pmap = {"fra_Latn": p}
    for i in range(8):
        print(f"{i} => {p(i)}")
    p_inv = p.get_inverse()
    for i in range(8):
        print(f"{i} => {p_inv(i)}")
    save_permutation_map(pmap, 'foo.json')
    pmap2 = load_permutation_map('foo.json')
    q = pmap2["fra_Latn"]
    q_inv = q.get_inverse()
    for i in range(8):
        print(f"{i} => {q(i)}")
    for i in range(8):
        print(f"{i} => {q_inv(i)}")
