import numpy as np
from collections import defaultdict
import math

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import pandas as pd
import matplotlib.pyplot as plt

import time
import multiprocessing 
from pathlib import Path

from sklearn.decomposition import PCA
import networkx as nx
import regex as re
from itertools import combinations
from datetime import datetime
from scipy.stats import fisher_exact

remap = {
    # Bread/Bakery
    'rolls/buns': '*bread',
    'roll products ': '*bread',
    'brown bread': '*bread',
    'white bread': '*bread',
    'zwieback': '*bread',
    'semi-finished bread': '*bread',
    'long life bakery product': '*bread',
    'pastry': '*bread',
    'cake bar': '*bread',
    'waffles': '*bread',
    
    # Dairy - Milk
    'whole milk': '*milk',
    'UHT-milk': '*milk',
    'butter milk': '*milk',
    'condensed milk': '*milk',
    
    # Dairy - Cheese
    'hard cheese': '*cheese',
    'sliced cheese': '*cheese',
    'cream cheese ': '*cheese',
    'spread cheese': '*cheese',
    'processed cheese': '*cheese',
    'curd cheese': '*cheese',
    'soft cheese': '*cheese',
    'specialty cheese': '*cheese',
    
    # Dairy - Other
    'curd': '*yogurt',
    'whipped/sour cream': '*cream',
    
    # Meat
    'beef': '*meat',
    'pork': '*meat',
    'chicken': '*meat',
    'turkey': '*meat',
    'hamburger meat': '*meat',
    
    # Processed Meat
    'sausage': '*processed meat',
    'frankfurter': '*processed meat',
    'ham': '*processed meat',
    'liver loaf': '*processed meat',
    'meat spreads': '*processed meat',
    
    # Fish
    # 'canned fish': '*fish',
    # 'frozen fish': '*fish',
    
    # Frozen Foods
    'frozen vegetables': '*frozen',
    'frozen dessert': '*frozen',
    'frozen meals': '*frozen',
    'frozen potato products': '*frozen',
    'frozen fruits': '*frozen',
    
    # Vegetables
    'other vegetables': '*vegetables',
    'packaged fruit/vegetables': '*vegetables',
    'root vegetables': '*vegetables',
    'onions': '*vegetables',
    'pickled vegetables': '*vegetables',
    'canned vegetables': '*vegetables',
    
    # Fruit
    'tropical fruit': '*fruit',
    'pip fruit': '*fruit',
    'grapes': '*fruit',
    'berries': '*fruit',
    'citrus fruit': '*fruit',
    'canned fruit': '*fruit',
    
    # Beverages - Alcohol
    'bottled beer': '*alcohol',
    'canned beer': '*alcohol',
    'white wine': '*alcohol',
    'red/blush wine': '*alcohol',
    'sparkling wine': '*alcohol',
    'prosecco': '*alcohol',
    'liquor': '*alcohol',
    'liquor (appetizer)': '*alcohol',
    'whisky': '*alcohol',
    'brandy': '*alcohol',
    'rum': '*alcohol',
    
    # Beverages - Non-Alcohol
    'soda': '*beverages',
    'bottled water': '*beverages',
    'fruit/vegetable juice': '*beverages',
    'misc. beverages': '*beverages',
    'cocoa drinks': '*beverages',
    
    # Snacks
    'salty snack': '*snacks',
    'chocolate': '*snacks',
    'candy': '*snacks',
    'chewing gum': '*snacks',
    'specialty chocolate': '*snacks',
    'chocolate marshmallow': '*snacks',
    'nut snack': '*snacks',
    'nuts/prunes': '*snacks',
    'popcorn': '*snacks',
    'specialty bar': '*snacks',
    'snack products': '*snacks',
    
    # Condiments
    'mayonnaise': '*condiments',
    'ketchup': '*condiments',
    'mustard': '*condiments',
    'vinegar': '*condiments',
    'sauces': '*condiments',
    'salad dressing': '*condiments',
    
    # Baking/Cooking
    'flour': '*baking',
    'sugar': '*baking',
    'baking powder': '*baking',
    'salt': '*baking',
    'spices': '*baking',
    'herbs': '*baking',
    
    # Spreads
    'sweet spreads': '*spreads',
    'jam': '*spreads',
    'honey': '*spreads',
    'margarine': '*spreads',
    'butter': '*spreads',
    
    # Cleaning
    'detergent': '*cleaning',
    'cleaner': '*cleaning',
    'abrasive cleaner': '*cleaning',
    'dish cleaner': '*cleaning',
    'softener': '*cleaning',
    'soap': '*cleaning',
    
    # Household
    'house keeping products': '*household',
    'napkins': '*household',
    'kitchen towels': '*household',
    'cling film/bags': '*household',
    'shopping bags': '*household',
    'light bulbs': '*household',
    'candles': '*household',
    'dishes': '*household',
    'cookware': '*household',
    
    # Personal Care
    'hygiene articles': '*personal care',
    'dental care': '*personal care',
    'male cosmetics': '*personal care',
    'female sanitary products': '*personal care',
    'make up remover': '*personal care',
    'baby cosmetics': '*personal care',
    
    # Prepared Foods
    'finished products': '*prepared foods',
    'Instant food products': '*prepared foods',
    'instant coffee': '*prepared foods',
    'pasta': '*prepared foods',
    'rice': '*prepared foods',
    'cereals': '*prepared foods',
    'soups': '*prepared foods',
    'potato products': '*prepared foods',
    
    # Pet Care
    'cat food': '*pet care',
    'dog food': '*pet care',
    'tidbits': '*pet care',
    
    # Organic
    'organic products': '*organic',
    'organic sausage': '*organic',
    
    # Desserts
    'ice cream': '*desserts',
    'dessert': '*desserts',
    'frozen dessert': '*desserts',
    'dessert': '*desserts'

}
# remap = {}


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

def gen_set(vertices, v):
    setset = []
    indices = list(range(v))
    while True:
        itemset = []
        for i in indices:
            itemset.append(vertices[i])
        setset.append(itemset)
        i = v - 1
        while i >= 0:
            if indices[i] < len(vertices) - v + i:
                break
            i -= 1
        if i < 0:
            break
        indices[i] += 1
        for j in range(i + 1, v):
            indices[j] = indices[j - 1] + 1
    return setset

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

def gen_cooccurrence_matrix(data):
    df = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i])):
                if k != j:
                    if data[i][j] not in df:
                        df[data[i][j]] = {}
                    if data[i][k] not in df[data[i][j]]:
                        df[data[i][j]][data[i][k]] = 1
                    else:
                        df[data[i][j]][data[i][k]] += 1
    keys = list(df.keys())
    for outer_key in keys:
        for inner_key in keys:
            if inner_key not in df[outer_key]:
                df[outer_key][inner_key] = 0

    return pd.DataFrame(df)


def xclat_broken(N, diffset, tidset, keys, global_keys, threshold, mode, v = 0):

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
        results.update(xclat(N, diffset, tidset, next_keys, global_keys, threshold, mode, v = v + 1))
    else:
        global_keys.remove(next_keys[0])
        results.update(xclat(N, diffset, tidset, global_keys.copy(), global_keys.copy(), threshold, mode, v = 0))
    return results

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


def eclat_old(tidset, keys, threshold, avoid, solution, x, v):
    def compute_support(intersection):
        return intersection / N
    
    results = []
    
    if x == len(keys):
        return results


    if v == 0:
        key = (keys[x],)
        if key not in solution:
            for h in range(len(keys)):
                solution[(keys[h],)] = compute_support(len(tidset[keys[h]]))

        support = solution[key]
        if support > threshold:
            # print(keys[x], support)
            [results.append(result) for result in eclat(tidset, keys, threshold, avoid, solution, x, v + 1)]
    else:
        indices = list(range(v))
        successes = 0
        iterations = 0
        while True:
            itemset = []
            go = True
            for i in indices:
                if i != x:
                    itemset.append(keys[i])
                    for start in range(len(itemset)):
                        subset = tuple(itemset[start:])
                        iterations += 1
                        if subset in avoid:
                            go = False
                            break
            if len(itemset) == v:
                if go == True:


                    
                    soln_key = tuple(sorted([keys[x]] + itemset))

                    if soln_key in solution:
                        support = solution[soln_key]
                    else:
                        intersection = set(tidset[keys[x]])
                        for key in itemset:
                            intersection.intersection_update(tidset[key])

                        support = compute_support(len(intersection))
                    if support > threshold:
                        # print(keys[x], itemset, support)
                        successes += 1
                        if soln_key not in solution:
                            solution[soln_key] = support

                        target_key = (keys[x],)

                        conf = support / solution[target_key]

                        itemset_key = tuple(itemset)
                        if itemset_key in solution:
                            itemset_supp = solution[itemset_key]
                        else:
                            
                            itemset_intersection = set(tidset[itemset[0]])
                            for key in itemset[1:]:
                                itemset_intersection.intersection_update(tidset[key])
                            itemset_supp = compute_support(len(itemset_intersection))
                            solution[itemset_key] = itemset_supp
                        
                        lift = support / (solution[target_key] * itemset_supp)
                        conv = (1 - itemset_supp) / (1 - conf)
                        lev = support - solution[target_key] * itemset_supp

                        result_dict = {f"{keys[x]} -> {', '.join(itemset)}": 
                                       {"supp":support,
                                        "conf": conf,
                                        "lift": lift,
                                        "conv": conv,
                                        "lev": lev,
                                        }}
                        results.append(result_dict)
                    else:
                        avoid.add(tuple(itemset))
                        # print("failed", keys[x], itemset, support)
                
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



        if successes > 0:
            [results.append(result) for result in eclat(tidset, keys, threshold, avoid, solution, x, v + 1)]
        else:
            [results.append(result) for result in eclat(tidset, keys, threshold, set(), solution, x + 1, 0)]
        

    return results


def read_csv(date_range, key_label):
    df = pd.read_csv(Path(__file__).parent / "Groceries_dataset.csv")

    data = {}
    unique = set()
    for i in range(len(df)):
        row = dict(df.iloc[i])
        date = datetime.strptime(row["Date"], '%d-%m-%Y')
        # key = f"{date.timetuple().tm_yday}-{date.isocalendar()[0]}" #day/year
        # key = f"{row[key_label]}-{date.isocalendar()[1]}-{date.isocalendar()[0]}" #week/year
        # key = f"{row[key_label]}-{date.month}-{date.isocalendar()[0]}" #month/year
        # key = f"{row[key_label]}-{date.month % 4}-{date.isocalendar()[0]}" #season/year
        # key = f"{row[key_label]}-{date.isocalendar()[0]}" #year
        key = row[key_label] #none
        
        month = int(re.search(r'-(\d+)-', row["Date"]).group(1))
        if month in date_range:
            items = [row["itemDescription"]]
            if items[0] in remap:
                items.append(remap[items[0]])
            
            for item in items:
                if item not in unique:
                    unique.add(item)
                if key in data:
                    if item not in data[key]:
                        data[key].append(item)
                else:
                    data[key] = []

    return [data[key] for key in list(data.keys())]



def plot_performance_data(data, title='Eclat/Declat Execution Time Ratio Analysis', 
                         figsize=(12, 8), style='both'):
    """
    Plot performance data showing execution time ratios across different samples and categories
    
    Parameters:
    data: list of dictionaries with performance data
    title: plot title
    figsize: figure size tuple (width, height)
    style: 'line', 'scatter', or 'both'
    """
    
    # Parse the data and organize by category count
    category_data = defaultdict(lambda: {'samples': [], 'ratios': []})
    
    for entry in data:
        # Extract key and parse sample count
        key = list(entry.keys())[0]
        sample_count = int(key)
        
        # Process each tuple in the value list
        for ratio, categories in entry[key]:
            category_data[categories]['samples'].append(sample_count)
            category_data[categories]['ratios'].append(ratio)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Colors and markers for different categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot data for each category
    for i, (cat_count, cat_data) in enumerate(sorted(category_data.items())):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Sort by sample count for cleaner lines
        sorted_pairs = sorted(zip(cat_data['samples'], cat_data['ratios']))
        samples, ratios = zip(*sorted_pairs)
        
        label = f'{cat_count} categories'
        
        if style in ['line', 'both']:
            plt.plot(samples, ratios, color=color, linewidth=2, 
                    alpha=0.7, label=label if style == 'line' else None)
        
        if style in ['scatter', 'both']:
            plt.scatter(samples, ratios, color=color, marker=marker, 
                       s=60, alpha=0.8, edgecolors='white', linewidth=1,
                       label=label if style == 'scatter' else label)
    
    # Customize the plot
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Execution Time Ratio (Eclat/Declat)', fontsize=12)
    
    # Add horizontal line at ratio = 1 for reference
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, 
                label='Equal Performance (Ratio = 1)')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add some statistics text
    all_ratios = []
    for cat_data in category_data.values():
        all_ratios.extend(cat_data['ratios'])
    
    avg_ratio = np.mean(all_ratios)
    plt.text(0.02, 0.98, f'Overall Avg Ratio: {avg_ratio:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Performance Summary ===")
    for cat_count, cat_data in sorted(category_data.items()):
        ratios = cat_data['ratios']
        print(f"{cat_count} categories:")
        print(f"  Average ratio: {np.mean(ratios):.3f}")
        print(f"  Min ratio: {np.min(ratios):.3f}")
        print(f"  Max ratio: {np.max(ratios):.3f}")
        print(f"  Std deviation: {np.std(ratios):.3f}")
    
    return category_data
        


def main():
    key_label = "Member_number"
    # date_ranges = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    # date_ranges = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
    # date_ranges = [[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 1, 2],]
    date_ranges = [[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]]
    # date_ranges = [[12, 1, 2]]
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    deltas = {}
    for s in range(len(date_ranges)):
        print(s) # print(seasons[s])
        data = read_csv(date_ranges[s], key_label)

        lengths = []
        for member in data:
            lengths.append(len(member))

        mean = np.mean(lengths)
        print(mean)

        N = len(data)
        print(N)
        encoded = oh_encode(data)
        coocm = gen_cooccurrence_matrix(data)
        tidset = gen_tidset(data)
        diffset = gen_diffset(data)
        keys = list(tidset.keys())

        # supps = xclat(diffset, tidset, keys.copy(), keys.copy(), N * .05, "ec")
        # rules = compute_rules(N, supps)
        # rules = sorted(rules.items(), key=lambda x: x[1]['conf'], reverse=True)
        
        # for rule in rules[:10]:
        #     print(rule)
        # input()


    if True:
        # # Function to prepare data for plotting
        # def prepare_data(data_dict):
        #     sizes = [len(data_dict[item]) for item in data_dict.keys()]
        #     return sorted(sizes, reverse=True)

        # # Prepare data for both plots
        # tidset_sizes = prepare_data(tidset)
        # diffset_sizes = prepare_data(diffset)

        # # Create figure with two subplots
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # # Plot 1: Tidset
        # x1 = range(len(tidset_sizes))
        # ax1.fill_between(x1, tidset_sizes, alpha=0.6, color='steelblue', edgecolor='darkblue', linewidth=2)
        # ax1.plot(x1, tidset_sizes, color='darkblue', linewidth=2)
        # ax1.set_xlabel('Items (sorted by size)', fontsize=12, fontweight='bold')
        # ax1.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        # ax1.set_title('Tidset', fontsize=14, fontweight='bold')
        # ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # # Plot 2: Diffset
        # x2 = range(len(diffset_sizes))
        # ax2.fill_between(x2, diffset_sizes, alpha=0.6, color='coral', edgecolor='darkred', linewidth=2)
        # ax2.plot(x2, diffset_sizes, color='darkred', linewidth=2)
        # ax2.set_xlabel('Items (sorted by size)', fontsize=12, fontweight='bold')
        # ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        # ax2.set_title('Diffset', fontsize=14, fontweight='bold')
        # ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # plt.tight_layout()
        # plt.show()

        # input()
        
        
        # for key in coocm.keys():
        #     print(key)


        keys = list(tidset.keys())
        # supps = xclat(tidset.copy(), tidset.copy(), keys.copy(), keys.copy(), 0.01, "ec")
        # rules = compute_rules(supps)

        # target_items = {"canned beer", "brown bread"}

        # for rule in rules:
        #     # if rule == (('*baking', 'sausage', 'root vegetables'), ('other vegetables',)):
        #     # if all(item in target_items for item in rule[0]):
        #     if ("canned beer", "brown bread") == rule[0]:
        #         print(rule, rules[rule])

        # Only look at rules meeting ALL thresholds
        # interesting = {
        #     k: v for k, v in rules.items()
        #     if v['lift'] >= 1.3 and v['supp'] >= 0.001 and v['kulc'] >= .4
        # }

        # Then sort by whatever seems most important
        # sorted_rules = list(sorted(interesting.items(), key=lambda item: item[1]['kulc'], reverse=True))
        # for rule in sorted_rules:
        #     print(rule)
        # input()

        # input()

        # to_find = ["*processed meat", "ice cream", "*alcohol"]
        # interest = []
        # for rule in rules:
        #     if all(item in rule[0] for item in ("*baking", "sausage")):
        #         interest.append({rule: rules[rule]})

        # sorted_interest = sorted(interest, key=lambda item: list(item.values())[0]['lift'], reverse=True)

        # for rule in sorted_interest:
        #     print(rule)

        # input()


        # rule_keys = ["supp", "conf", "lift", "conv"]

        # for rule in rules:
        #     # if isinstance(rule, str) is True:
        #     if True:
        #         if rule in deltas:
        #             for rule_key in rule_keys:
        #                 deltas[rule][rule_key].append(rules[rule][rule_key])
        #         else:
        #             deltas[rule] = {"supp": [rules[rule]["supp"]], 
        #                             "conf": [rules[rule]["conf"]], 
        #                             "lift": [rules[rule]["lift"]], 
        #                             "conv": [rules[rule]["conv"]], 
        #                             "lev": [rules[rule]["lev"]]}
        # #             deltas[rule] = {"supp": [0 for _ in range(s)] + [rules[rule]["supp"]], 
        # #                             "conf": [0 for _ in range(s)] + [rules[rule]["conf"]], 
        # #                             "lift": [1.0 for _ in range(s)] + [rules[rule]["lift"]], 
        # #                             "conv": [1.0 for _ in range(s)] + [rules[rule]["conv"]], 
        # #                             "lev": [0 for _ in range(s)] + [rules[rule]["lev"]]}
        # # for key in deltas:
        # #     if len(deltas[key]["supp"]) < s + 1:
        # #         deltas[key]["supp"].append(0)
        # #         deltas[key]["conf"].append(0)
        # #         deltas[key]["lift"].append(1.0)
        # #         deltas[key]["conv"].append(1.0)
        # #         deltas[key]["lev"].append(0.0)



        # pca = PCA()
        # X = pca.fit_transform(encoded)
        # cum_var = np.cumsum(pca.explained_variance_ratio_)
        # for i in range(len(cum_var)):
        #     print(i, cum_var[i])
        # loadings = pd.DataFrame(
        #     pca.components_,
        #     columns=encoded.columns,
        #     index=[f'PC{i+1}' for i in range(pca.n_components_)]
        # )
        # print(loadings.head())


    # results = {}
    # for key in deltas:
    #     mean = np.mean(deltas[key]["lift"])
    #     cv = np.std(deltas[key]["lift"]) / mean
    #     results[key] = {"mean": mean, "cvar": cv, "values": deltas[key]}

    # sorted_results = (sorted(results.items(), key=lambda x: x[1]['cvar'], reverse=True))

    # to_pop = []
    # for i in range(len(sorted_results)):
    #     antecedent = sorted_results[i][0][0] 
    #     consequent = sorted_results[i][0][1]

    #     for item in antecedent:
    #         if item in remap:
    #             if remap[item] in consequent:
    #                 to_pop.append(i)
    #                 break
    #             elif remap[item] in antecedent:
    #                 to_pop.append(i)
    #                 break
        
    #     for item in consequent:
    #         if item in remap:
    #             if remap[item] in antecedent:
    #                 to_pop.append(i)
    #                 break
    #             elif remap[item] in consequent:
    #                 to_pop.append(i)
    #                 break

    # for i in sorted(to_pop, reverse=True):
    #     sorted_results.pop(i)

    # print(len(sorted_results))

    # i = 0
    # for result in sorted_results:
    #     print(result)
    #     if i % 100 == 0:
    #         input()
    #     i += 1


        # G = nx.Graph()
        # for edge in edges:
        #     if edges[edge]:
        #         G.add_edge(edge[0], edge[1], weight = edges[edge])

        global_node_counts = {}
        global_node_totals = {}
        for item in coocm.columns:
            global_node_totals[item] = len(tidset[item]) / N
            global_node_counts[item] = len(tidset[item])


        minsup = 0.01
        min_pairwise_sup = .04
        minlift = 1.34
        maxlift = 1000

        graphs = []
        for i in range(len(coocm.columns)):
            for j in range(i + 1, len(coocm.columns)):
                item_i = coocm.columns[i]
                item_j = coocm.columns[j]
                go = True
                if item_i in remap:
                    if item_j == remap[item_i]:
                        go = False
                elif item_j in remap:
                    if item_i == remap[item_j]:
                        go = False
                elif item_i in remap and item_j in remap:
                    go = False

                # x = coocm.loc[item_i, item_j] / N #for support
                # if x > minsup:
                #     graphs[(item_i, item_j)] = x

                #for lift
                print(i, j)
                if go == True:
                    x = coocm.loc[item_i, item_j] * N / (len(tidset[coocm.columns[i]]) * len(tidset[coocm.columns[j]]))
                    #COSINE # x = coocm.loc[item_i, item_j] / (len(tidset[coocm.columns[i]]) * len(tidset[coocm.columns[j]]))
                    #KULC x = (coocm.loc[item_i, item_j] / len(tidset[coocm.columns[i]]) + coocm.loc[item_i, item_j] / len(tidset[coocm.columns[j]])) / 2
                    y = (coocm.loc[item_i, item_j]) / N
                    if x > minlift and x < maxlift and y > min_pairwise_sup:
                        placed = False
                        for k in range(len(graphs)):
                            # Check if item_i or item_j appears in any tuple key
                            if any(item_i in key or item_j in key for key in graphs[k].keys()):
                                graphs[k][(item_i, item_j)] = x
                                placed = True
                                break
                        if not placed:
                            graphs.append({(item_i, item_j): x})
                

        for graph in graphs:
            sorted_lifts = sorted(graph.items(), key=lambda x: x[1], reverse=True)
            all_lifts = [graph[key] for key in graph]
            mean = np.sum(all_lifts) / len(all_lifts)
            for_var = []
            for lift in all_lifts:
                for_var.append((lift - mean) ** 2)
            sd = np.sqrt(np.sum(for_var) / N - 1)
            ub = 3#2.0 * sd + mean

            vertices = []
            for edge in graph:
                for vertex in edge:
                    if vertex not in vertices:
                        vertices.append(vertex)
            
            graph_supps = xclat(tidset.copy(), tidset.copy(), vertices.copy(), vertices.copy(), minsup * N, "ec")
            graph_rules = compute_rules(N, graph_supps)
            sig_rules = []
            for rule in graph_rules:
                if len(rule[0]) + len(rule[1]) > 2:
                    sig_rules.append((rule, graph_rules[rule]))

            sorted_rules = sorted(sig_rules, key=lambda x: x[1]['conf'], reverse=False)
            print(len(sorted_rules))
            input()
            for rule in sorted_rules:
                print(rule)
            
            sig_rules_keys = set()
            for rule, metrics in sig_rules:
                antecedent, consequent = rule
                sig_rules_keys.add(antecedent)
                sig_rules_keys.add(consequent)

            F = nx.Graph()
            G = nx.Graph()
            for key in graph:
                G.add_edge(key[0], key[1], weight=graph[key])
                if key in sig_rules_keys or (key[1], key[0]) in sig_rules_keys:
                    print(key)
                    F.add_edge(key[0], key[1], weight=graph[key])

            visuals = [G, F]

            for graph_object in visuals:
                if len(graph_object.nodes) > 0:
                    '''RAW OBSERVATION COUNTS'''
                    node_sizes = [global_node_counts[node] for node in graph_object.nodes()]
                    node_totals = {}
                    i = 0
                    for node in graph_object.nodes():
                        node_totals[node] = node_sizes[i]
                        i += 1

                    '''EDGE WEIGHTS BASED ON CO-OCCURRENCE'''
                    # node_sizes = []
                    # node_totals = {}
                    # for node in G.nodes():
                    #     # Sum of all co-occurrences for this item
                    #     total_cooccur = coocm.loc[node].sum()
                    #     node_sizes.append(total_cooccur)
                    #     node_totals[node] = total_cooccur
                        
                    # Normalize for visualization
                    node_sizes = np.array(node_sizes)
                    node_sizes_scaled = (node_sizes / node_sizes.max()) * 3000 + 2000 

                    # Create a mapping of edges to their best (highest lift) rule
                    edge_to_rule = {}

                    for rule, metrics in sig_rules:
                        antecedent, consequent = rule
                        # Get all items involved in this rule
                        all_items = set(antecedent + consequent)
                        
                        # Find all edges between these items that exist in the graph
                        for item1 in all_items:
                            for item2 in all_items:
                                if item1 < item2:  # Avoid duplicates (since undirected graph)
                                    if graph_object.has_edge(item1, item2):
                                        edge = (item1, item2)
                                        # Keep the rule with highest lift for this edge
                                        if edge not in edge_to_rule or metrics['lift'] > edge_to_rule[edge]['lift']:
                                            edge_to_rule[edge] = metrics

                    # Draw
                    plt.figure(figsize=(9, 9))
                    # pos = nx.spring_layout(graph_object, k=2, iterations=50)
                    pos = nx.kamada_kawai_layout(G)

                    # Draw nodes
                    nx.draw_networkx_nodes(graph_object, pos, 
                                        node_color='lightblue', 
                                        node_size=node_sizes_scaled)

                    # Draw node labels (item names)
                    nx.draw_networkx_labels(graph_object, pos, font_size=12)

                    # Prepare edges with colors based on association rules
                    edges = graph_object.edges()
                    weights = [graph_object[u][v]['weight'] for u, v in edges]
                    max_weight = max(weights) if weights else 1

                    edge_colors = []
                    edge_widths = []

                    for u, v in edges:
                        weight = graph_object[u][v]['weight']
                        edge_widths.append(weight / max_weight * 5)
                        
                        # Check if this edge is part of any rule
                        edge_key = (u, v) if u < v else (v, u)
                        
                        if edge_key in edge_to_rule:
                            lift_val = edge_to_rule[edge_key]['lift']
                            if lift_val > 2.5:  # Very strong
                                edge_colors.append('red')
                            elif lift_val > 2.0:  # Strong
                                edge_colors.append('orange')
                            elif lift_val > 1.5:  # Moderate
                                edge_colors.append('gold')
                            else:  # Weak
                                edge_colors.append('lightgreen')
                        else:
                            edge_colors.append('gray')  # Not in any rule

                    # Draw edges with colors
                    nx.draw_networkx_edges(graph_object, pos, 
                                        width=edge_widths,
                                        edge_color=edge_colors)

                    # Edge labels
                    edge_labels = nx.get_edge_attributes(graph_object, 'weight')
                    edge_labels = {edge: round(edge_labels[edge], 3) for edge in edge_labels}
                    nx.draw_networkx_edge_labels(graph_object, pos, edge_labels, font_size=10)

                    # Add node totals above each node
                    pos_above = {node: (x, y + 0.08) for node, (x, y) in pos.items()}  # offset upward
                    node_total_labels = {node: round(total, 3) for node, total in node_totals.items()}
                    nx.draw_networkx_labels(graph_object, pos_above, node_total_labels, 
                                        font_size=10, font_color='blue', 
                                        bbox=dict(facecolor='white', edgecolor='none', alpha=1, boxstyle='round,pad=0.3'))

                    # Legend elements
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='blue', label='Item Support'),
                        Patch(facecolor='red', label='Lift > 2.5 (Very Strong)'),
                        Patch(facecolor='orange', label='Lift > 2.0 (Strong)'),
                        Patch(facecolor='gold', label='Lift > 1.5 (Moderate)'),
                        Patch(facecolor='lightgreen', label='Lift > 1 (Weak)'),
                        Patch(facecolor='gray', label='No Rule / Co-occurrence Only')
                    ]
                    plt.legend(handles=legend_elements, loc='upper right')

                    plt.title(f"Transient Co-occurrence Graph")
                    plt.axis('off')
                    plt.tight_layout()
                    if len(sig_rules) > 0:
                        plt.show()



        '''
        vertex_weights = {}
        for node in G.nodes():
            total_weight = sum(data['weight'] for u, v, data in G.edges(node, data=True))
            vertex_weights[node] = total_weight
            print(node, total_weight)

        # Convert to array for analysis
        weights_array = np.array(list(vertex_weights.values()))

        # Compute median and MAD
        median_weight = np.median(weights_array)
        mad = np.median(np.abs(weights_array - median_weight))

        print(f"Median vertex weight sum: {median_weight}")
        print(f"MAD: {mad}")

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(weights_array, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(median_weight, color='red', linestyle='--', linewidth=2, 
                    label=f'Median: {median_weight:.1f}')
        plt.axvline(median_weight + mad, color='orange', linestyle=':', linewidth=2, 
                    label=f'Median + MAD: {median_weight + mad:.1f}')
        plt.axvline(median_weight - mad, color='orange', linestyle=':', linewidth=2, 
                    label=f'Median - MAD: {median_weight - mad:.1f}')
        plt.xlabel('Sum of Edge Weights per Vertex')
        plt.ylabel('Frequency')
        plt.title('Distribution of Vertex Weight Sums')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Optional: see which nodes are outliers
        print("\nTop 5 vertices by total edge weight:")
        for node, weight in sorted(vertex_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{node}: {weight}")
        '''


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


    # frequent_itemsets = apriori(encoded, min_support=0.01, use_colnames=True)
    # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    # print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']])


    # names = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # D = len(names)
    # N = 10000
    # trend = [np.random.random() for name in names]

    # data = gen_data(N, D)
    # diffset = gen_diffset(data)
    # tidset = gen_tidset(data)

    # def test_xclat():
    #     keys = sorted(diffset.keys())
    #     supps = xclat(diffset, tidset, keys.copy(), keys.copy(), 0.1, "de")
    #     rules = compute_rules(supps)
    #     sorted_rules = sorted(rules.items(), key=lambda item: item[1]['conf'], reverse=True)
    #     for rule in sorted_rules[:10]:
    #         print(rule)

    #     encoded = oh_encode(data)
    #     frequent_itemsets = apriori(encoded, min_support=0.1, use_colnames=True)
    #     rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.1)
    #     top_rules = rules.sort_values('confidence', ascending=False).head(10)

    #     for index, rule in top_rules.iterrows():
    #         antecedent = list(rule['antecedents'])
    #         consequent = list(rule['consequents'])
            
    #         print(f"{antecedent} -> {consequent}")
    #         print(f"  Confidence: {rule['confidence']:.4f}")
    #         print(f"  Support: {rule['support']:.4f}")
    #         print(f"  Lift: {rule['lift']:.4f}")
    #         print()


def runtimes():
    import tracemalloc
    data = read_csv([i for i in range(13)], "Member_number")
    diffset = gen_diffset(data)
    tidset = gen_tidset(data)
    N = len(data)
    D = len(list(diffset.keys()))
    print(N, D)
    results = []
    run = 0
    thresholds = [.2, .1,]
    for t in thresholds:
        keys = list(diffset.keys())
        start = time.time()
        tracemalloc.start()
        supps = xclat(diffset, tidset, keys.copy(), keys.copy(), round(t * N), mode="ec")
        current, peak_memory = tracemalloc.get_traced_memory()
        end = time.time()
        tracemalloc.stop()
        print(t, len(supps), end-start, peak_memory / 1024 ** 2)
        results.append({"Time": end-start, "threshold": t, "n_itsets": len(supps)})
    
    thresholds = [result["threshold"] for result in results]
    times = [result["Time"] for result in results]
    n_itsets = [result["n_itsets"] for result in results]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Time vs Threshold
    ax1.plot(thresholds, times, marker='o', linewidth=2, markersize=6, color='blue')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('DEclat Execution Time vs Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Number of Rules vs Threshold
    ax2.plot(thresholds, n_itsets, marker='s', linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Number of Itemsets', fontsize=12)
    ax2.set_title('DEclat Number of Itemsets vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    def runruntimes():
        runtimes()

    runruntimes()
    
    main()