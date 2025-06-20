from Bitext import Bitext
import random


text_files1 = {'l1': 'lang1.txt','l2': 'lang2.txt', 'l3': 'lang3.txt'}


class MixtureOfBitexts:
   def __init__ (self, text_files, sampling_probs, batch_size):
      self.batch_size = batch_size
      self.text_files = text_files
      self.sampling_probs = sampling_probs
      index_dict = {}
      for filenames_pair in sampling_probs.keys():
         index_dict[filenames_pair] = 0
      self.tracking = index_dict

   @classmethod
   def create_from_files(cls, text_files, sampling_probs, batch_size):
      return cls(text_files, sampling_probs, batch_size)
   

   def next_batch(self):
      random_number = random.random()
      acc = 0
      for pair, prob in self.sampling_probs.items():
         acc += prob
         if random_number < acc:
            el1 = pair[0]
            el2 = pair[1]
            pre_result1 = []
            pre_result2 = []
            list_result = []
            with open(self.text_files[el1], "r", encoding="utf-8") as f1, open(self.text_files[el2], "r", encoding="utf-8") as f2:
                 lines1 = f1.read().splitlines()
                 lines2 = f2.read().splitlines()
                 pair_list = list(zip(lines1, lines2))
                 pair_index = self.tracking[pair]
                 for i in range(self.batch_size):
                  pre_result1.append(pair_list[(i+pair_index)%len(pair_list)][0])
                  pre_result2.append(pair_list[(i+pair_index)%len(pair_list)][1])
                 self.tracking[pair] += self.batch_size
                 list_result.append(tuple(pre_result1))
                 list_result.append(tuple(pre_result2))
                 list_result.append(el1)
                 list_result.append(el2)
                 print(tuple(list_result))
                 return tuple(list_result)
                 


mix = MixtureOfBitexts.create_from_files(text_files=text_files1, sampling_probs={('l1', 'l2'): 0.8, ('l1', 'l3'): 0.2}, batch_size=2)

mix.next_batch()
mix.next_batch()
mix.next_batch()
mix.next_batch()
mix.next_batch()
mix.next_batch()
mix.next_batch()
mix.next_batch()