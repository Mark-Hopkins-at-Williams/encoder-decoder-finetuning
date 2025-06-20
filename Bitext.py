import torch
from pathlib import Path


DATA_DIR   = Path(".")   

class Bitext(torch.utils.data.IterableDataset):
    def __init__(self, file1, file2): # file1 and file 2 are names of the files used 
        super(Bitext).__init__()
        self.f1 = file1
        self.f2 = file2
        self.f1path = DATA_DIR / file1
        self.f2path = DATA_DIR / file2
        
    def __iter__(self):
        with open(self.f1, "r", encoding="utf-8") as f1, open(self.f2, "r", encoding="utf-8") as f2:
          lines1 = f1.read().splitlines()
          lines2 = f2.read().splitlines()
          pair_list = list(zip(lines1, lines2))
          return iter(pair_list)


          
          
          

