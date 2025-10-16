import numpy as np
from collections import defaultdict
import math

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt

import time
import multiprocessing 
from pathlib import Path


def gen_data(n, d):
    data = []
    for i in range(n):
        cart = []
        for j in range(d):
            x = np.random.binomial(1, trend[j], 1)
            if x == 1:
                cart.append(names[j])
        data.append(cart.copy())
    return data

def gen_diffset(data):

    n = len(data)

    tidset = defaultdict(set)
    for i in range(n):
        for j in range(len(data[i])):
            tidset[data[i][j]].add(i)
    
    diffset = defaultdict(set)
    keys = sorted(tidset.keys())
    for key in keys:
        for i in range(n):
            if i not in tidset[key]:
                diffset[key].add(i)
    return diffset

def gen_tidset(data):

    tidset = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in tidset:
                tidset[data[i][j]] = set()
            tidset[data[i][j]].add(i)
    return tidset


def oh_encode(data):
    df = []
    for i in range(len(data)):
        df.append({label: 1 for label in data[i]})
    df = pd.DataFrame(df).fillna(0).astype(int)
    return df


def xclat(diffset, tidset, keys, global_keys, threshold, mode, v = 0):

    def compute_support_diffset(union):
        return (N - union) / N
    
    def compute_support_tidset(intersection):
        return intersection / N
    
    def dynamic_support(tset):
        if len(diffset[tset[0]]) > len(tidset[tset[0]]):
            support = compute_support_tidset(len(get_intersection(tset)))
        else:
            support = compute_support_diffset(len(get_union(tset)))
        return support
    
    def gen_set(v):
        setset = []
        indices = list(range(1, v + 1))
        while True:
            itemset = [keys[0]]
            go = True
            for i in indices:
                itemset.append(keys[i])
            setset.append(itemset)
            i = v - 1
            while i >= 0:
                if indices[i] < len(keys) - v + i:
                    break
                i -= 1
            if i < 0:
                break
            indices[i] += 1
            for j in range(i + 1, v):
                indices[j] = indices[j - 1] + 1
        return setset
    
    def get_union(tset):
        union = set()
        for key in tset:
            union.update(diffset[key])
        return union
    
    def get_intersection(tset):
        intersection = set(tidset[tset[0]])
        for key in tset[1:]:
            intersection.intersection_update(tidset[key])
        return intersection

    results = {}

    if len(keys) == 0:
        return results

    if v < len(keys):
        tset = gen_set(v)
    else:
        tset = keys.copy()
    elim = set()
    for i in range(len(tset)):
        if mode == "de":
            support = compute_support_diffset(len(get_union(tset[i])))
        elif mode == "ec":
            support = compute_support_tidset(len(get_intersection(tset[i])))
        elif mode == "multi":
            support = dynamic_support(tset[i])
        if support < threshold:
            if len(tset[i]) > 0:
                for value in tset[i][1:]:
                    if value not in elim:
                        elim.add(value)
            else:
                elim.add(tset[i][0])
        else:
            results[tuple(tset[i])] = support
    
    next_keys = [key for key in keys if key not in elim]
    
    if v < len(next_keys) - 1:
        results.update(xclat(diffset, tidset, next_keys, global_keys, threshold, mode, v = v + 1))
    else:
        global_keys.remove(next_keys[0])
        results.update(xclat(diffset, tidset, global_keys.copy(), global_keys.copy(), threshold, mode, v = 0))
    return results

def compute_rules(supps):

    def gen_set(v):
        setset = []
        indices = list(range(v))
        while True:
            itemset = []
            for i in indices:
                itemset.append(key[i])
            setset.append(itemset)
            i = v - 1
            while i >= 0:
                if indices[i] < len(key) - v + i:
                    break
                i -= 1
            if i < 0:
                break
            indices[i] += 1
            for j in range(i + 1, v):
                indices[j] = indices[j - 1] + 1
        return setset


    rules = {}
    for key in supps:
        if len(key) > 1:
            v = len(key) - 1
            ants = []
            for k in range(1, v + 1):
                sets = gen_set(k)
                ants += sets
            for ant in ants:
                coq_key = tuple(item for item in key if item not in ant)
                ant_key = tuple(ant)
                
                ant_supp = supps[ant_key]
                coq_supp = supps[coq_key]
                set_supp = supps[key]
                exjt = ant_supp * coq_supp
                

                conf = set_supp / ant_supp
                lift = set_supp / exjt
                conv = (1 - coq_supp) / (1 - conf) if conf < 1 else np.inf
                lev = set_supp - exjt

                rules.update({f"{', '.join(ant_key)} -> {', '.join(coq_key)}": 
                                        {"supp": set_supp,
                                            "conf": conf,
                                            "lift": lift,
                                            "conv": conv,
                                            "lev": lev,
                                            }})
                
                
        else:
            rules[key[0]] = {"support": supps[key], 
                             "conf": supps[key], 
                             "lift": 1.0, 
                             "conv": 1.0, 
                             "lev": 0.0}
            
    return rules

names = [
    "drywall", "screws", "nails", "sealant", "boards", "plywood", "insulation", "foam",
    "lumber", "2x4s", "2x6s", "2x8s", "studs", "joists", "beams", "posts",
    "cement", "concrete", "mortar", "grout", "rebar", "wire_mesh", "aggregate",
    "shingles", "roofing_tiles", "tar_paper", "flashing", "gutters", "downspouts",
    "paint", "primer", "brushes", "rollers", "sandpaper", "wood_stain", "varnish",
    "caulk", "adhesive", "glue", "epoxy", "silicone", "weatherstripping", "tape",
    "electrical_wire", "outlets", "switches", "breakers", "conduit", "junction_boxes",
    "pipes", "fittings", "valves", "fixtures", "faucets", "toilets", "sinks",
    "tiles", "flooring", "carpet", "hardwood", "laminate", "vinyl", "subflooring",
    "bolts", "washers", "nuts", "anchors", "brackets", "hinges", "handles",
    "windows", "doors", "frames", "trim", "molding", "baseboards", "crown_molding",
    "fiberglass_insulation", "spray_foam", "rigid_foam", "vapor_barrier", "house_wrap",
    "brick", "stone", "stucco", "siding", "aluminum_siding", "vinyl_siding",
    "tools", "hammer", "saw", "drill", "level", "measuring_tape", "square",
    "safety_glasses", "gloves", "hard_hat", "masks", "knee_pads", "work_boots",
    "ladder", "scaffolding", "tarps", "plastic_sheeting", "drop_cloths",
    "extension_cords", "work_lights", "generators", "compressors", "hoses"
]

# trend = [np.random.random() for name in names]

# times = []
# i = 0
# for n in range(10000, 20001, 3000):
#     row = []
#     for d in range(15, 32, 2):
#         N = n
#         data = gen_data(n, d)
#         encoded = oh_encode(data)
#         diffset = gen_diffset(data)
#         tidset = gen_tidset(data)
#         start = time.time()
#         keys = list(diffset.keys())
#         rules0 = xclat(diffset, tidset, keys.copy(), keys.copy(), 0.1, "de")
#         end1 = time.time()
#         rules1 = xclat(diffset, tidset, keys.copy(), keys.copy(), 0.1, "ec")
#         end2 = time.time()
#         row.append(((end2 - end1) / (end1 - start), d))
#         i += 1
#         print(i)
#     times.append({f"{n}": row})

# plot_performance_data(times)
# input()