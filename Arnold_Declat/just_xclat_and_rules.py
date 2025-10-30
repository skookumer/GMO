from itertools import combinations
from scipy.stats import fisher_exact

remap = {}

def xclat(diffset, tidset, keys, global_keys, threshold, mode, v = 0, max_v = -1, results = {}):
    # print(N)
    # input()

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
    
    def gen_set_itertools(v):
        return [[keys[0]] + list(combo) for combo in combinations(keys[1:], v)]

    
    def get_union(tset):
        union = set()
        for key in tset:
            union.update(diffset[key])
        return union
    
    def get_intersection(tset):
        sorted_keys = sorted(tset, key=lambda k: len(tidset[k]))
        intersection = set(tidset[sorted_keys[0]])
        
        for key in sorted_keys[1:]:
            intersection &= tidset[key]
        
        return len(intersection)
    
    def get_conditional_intersection(tset):
        if (tset[0],) in results:
            supp_x = results[(tset[0],)]
            diff_x = set(diffset[tset[0]])
            diff_all = set.union(*(diffset[key] - diff_x for key in tset[1:]))
            return supp_x - len(diff_all)
        else:
            return len(tidset[tset[0]])

    if len(keys) == 0 or v == max_v:
        return results

    if v < len(keys):
        tset = gen_set_itertools(v)
    else:
        tset = [[key] for key in keys]

    elim = set()
    passed = set()
    anchor = True

    if len(tset) == 1 and v == 0:
        support = len(tidset[tset[0][0]])
        if support >= threshold:
            results[(tset[0][0],)] = support
        else:
            anchor = False
    else:
        for i in range(len(tset)):
            if mode == "de":
                support = get_conditional_intersection(tset[i])
            elif mode == "ec":
                support = get_intersection(tset[i])
            # elif mode == "multi":
            #     support = dynamic_support(tset[i])
    
            if support < threshold:
                if len(tset[i]) > 1:
                    for value in tset[i][1:]:
                        elim.add(value)
                elif len(tset[i]) == 1:
                    elim.add(tset[i][0])
            else:
                results[tuple(tset[i])] = support
                for item in tset[i]:
                    passed.add(item)

    elim = {item for item in elim if item not in passed}

    next_keys = [key for key in keys if key not in elim]
    
    if v < len(next_keys) - 1 and anchor == True:
        results.update(xclat(diffset, tidset, next_keys, global_keys, threshold, mode, v = v + 1, max_v=max_v, results = results))
    elif len(next_keys) > 0:
        global_keys.remove(next_keys[0])
        results.update(xclat(diffset, tidset, global_keys.copy(), global_keys.copy(), threshold, mode, v = 0, max_v=max_v))
    return results


def compute_rules(N, supps):

    def get_maximal_itemsets():
        maximal = {}
        itemsets_sorted = sorted(supps.keys(), key=len, reverse=True)
        
        for itemset in itemsets_sorted:
            is_subset = any(set(itemset).issubset(set(maximal_set)) 
                        for maximal_set in maximal)
            if not is_subset:
                maximal[itemset] = supps[itemset]
        
        return maximal
    
    def get_closed_itemsets():
        closed = {}
        itemsets_sorted = sorted(supps.keys(), key=len, reverse=True)
        
        for itemset in itemsets_sorted:
            # Check if any superset has the same support
            has_superset_same_support = any(
                set(itemset).issubset(set(other_itemset)) and 
                supps[itemset] == supps[other_itemset]
                for other_itemset in supps 
                if len(other_itemset) > len(itemset)
            )
            
            if not has_superset_same_support:
                closed[itemset] = supps[itemset]
        
        return closed

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
    
    # good = get_maximal_itemsets()
    # good = get_closed_itemsets()

    rules = {}
    for key in supps:
        if len(key) > 1: #nd key in good:
            v = len(key) - 1
            ants = []
            for k in range(1, v + 1):
                sets = gen_set(k)
                ants += sets
            for ant in ants:
                try:
                    coq_key = tuple(item for item in key if item not in ant)
                    ant_key = tuple(ant)

                    all_items_in_rule = set(ant_key + coq_key)
                    has_redundancy = False
                    for item in all_items_in_rule:
                        if item in remap:
                            if remap[item] in all_items_in_rule:
                                has_redundancy = True
                                break

                    if has_redundancy:
                        break
                    
                    ant_supp = supps[ant_key] / N
                    coq_supp = supps[coq_key] / N
                    set_supp = supps[key] / N
                    exjt = ant_supp * coq_supp


                    # Fisher's exact test - build contingency table
                    # Counts (not probabilities)
                    n11 = supps[key]  # both ant and coq
                    n10 = supps[ant_key] - n11  # ant but not coq
                    n01 = supps[coq_key] - n11  # coq but not ant
                    n00 = N - n11 - n10 - n01  # neither
                    
                    contingency_table = [[n11, n10],
                                        [n01, n00]]
                    
                    oddsratio, pvalue = fisher_exact(contingency_table, alternative='greater')
                    
                    

                    conf = set_supp / ant_supp
                    lift = set_supp / exjt
                    conv = (1 - coq_supp) / (1 - conf) if conf < 1 else np.inf
                    lev = set_supp - exjt
                    cos = set_supp / np.sqrt(exjt)
                    kulc = (set_supp / ant_supp + set_supp / coq_supp) / 2

                    rules.update({(ant_key, coq_key): 
                                            {"supp": set_supp,
                                                "conf": conf,
                                                "lift": lift,
                                                "conv": conv,
                                                "lev": lev,
                                                "cos": cos,
                                                "kulc": kulc,
                                                "fisher_pvalue": pvalue,
                                                "fisher_odds": oddsratio
                                                }})
                except:
                    pass
                
                
        # else:
        #     rules[key[0]] = {"supp": supps[key] / N, 
        #                      "conf": supps[key] / N, 
        #                      "lift": 0.0, 
        #                      "conv": 0.0, 
        #                      "lev": 0.0,
        #                      "cos": 0.0,
        #                      "kulc": 0.5 * supps[key] / N}

    return rules