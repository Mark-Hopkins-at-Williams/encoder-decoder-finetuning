import random
import json

class create_random_permutation_with_fixed_points:
    def __init__(self, num:int, fixed_points):
        self.num = num
        self.fixed_points = fixed_points
        non_fixed_points = [i for i in list(range(num)) if i not in fixed_points]
        self.result_dict = {}
        self.inverse_dict = {}
        for i in range(num):
            if i in fixed_points:
                self.result_dict[i] = i
                self.inverse_dict[i] = i
            else:
                self.result_dict[i] = random.choice(non_fixed_points)
                non_fixed_points.remove(self.result_dict[i])
                self.inverse_dict[self.result_dict[i]] = i        
        
    def __call__(self, num):
        return self.result_dict[num]
    
    def get_inverse(self):
        q = create_random_permutation_with_fixed_points(self.num, self.fixed_points)
        q.result_dict = self.inverse_dict
        q.inverse_dict = self.result_dict
        return q
    

#function to save dictionary into json file
def save_permutation_map(pmap, filename):
    with open(filename, 'w') as f:
        all_data = {}
        for key,value in pmap.items():
            all_data[key] = {
                "fixed points": value.fixed_points,
                "rng": value.num,
                "result dictionary": value.result_dict,
                "inverse dictionary": value.inverse_dict
            }
        json.dump(all_data, f, indent=4)


#function to read json file
def load_permutation_map(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
        data = {}
        for key,value in data_dict.items():
            obj = create_random_permutation_with_fixed_points(
                num = value["rng"],
                fixed_points = value["fixed points"]
            )
            obj.result_dict = {int(i):s for i, s in value["result dictionary"].items()}
            obj.inverse_dict = {int(i):s for i, s in value["inverse dictionary"].items()}
            data[key] = obj
        return data

        
        
        



pmap = {"eng_Latn": create_random_permutation_with_fixed_points(8, [0, 1, 2, 7]),
        "fra_Latn": create_random_permutation_with_fixed_points(8, [0, 1, 2, 7])}
save_permutation_map(pmap, 'foo.json')
pmap2 = load_permutation_map('foo.json')
for i in range(8):
    print(f"{i} => {pmap2["eng_Latn"](i)}")