#!/usr/bin/env python
import numpy as np

def load_data():
   with open('restaurant.csv') as f:
      data = f.readlines()
      return data
   return None

def list_pairs(similarity=lambda x,y: 1):
   data = load_data()
   pairs_list = []
   for i in data:
      ci = i.split(',')[2:-1]
      for j in data:
          cj = j.split(',')[2:-1]
          pairs_list.append((similarity(ci,cj),
                              (i.split(',')[0],
                               j.split(',')[0]),
                             ci,cj))
   return pairs_list

def get_sample(N=10):
   pairs = list_pairs(jaccard)
   l = len(pairs)
   #indices_to_show = np.random.choice(range(0,l),N)
   
   #show hard examples & some random
   n = 0
   indices_to_show = []
   while n < N:
      i = np.random.choice(range(0,l),1)
      if i not in indices_to_show:
         if pairs[i][0] >= 0.2 and pairs[i][0] <= 0.8:
            indices_to_show.append(i)
            n += 1
         elif np.random.random <= 0.1:
            indices_to_show.append(i)
            n += 1
   '''
   #show easy examples & some random
   n = 0
   indices_to_show = []
   while n < N:
      i = np.random.choice(range(0,l),1)
      if i not in indices_to_show:
         if pairs[i][0] < 0.2 or pairs[i][0] > 0.8:
            indices_to_show.append(i)
            n += 1
         elif np.random.random <= 0.1:
            indices_to_show.append(i)
            n += 1
   '''
   return [pairs[i] for i in indices_to_show]

def jaccard(a,b):
   word_set_a = set(a.lower().split())
   word_set_b = set(b.lower().split())
   word_set_c = word_set_a.intersection(word_set_b)
   return float(len(word_set_c)) / (len(word_set_a) + len(word_set_b) - len(word_set_c)) 
