import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from collections import defaultdict
from itertools import combinations
from functools import reduce
import operator
import time, tracemalloc

## Helper Functions ## 
def sort_key(pair):
    X, T = pair
    # sort by support (ascending), then lexicographic item order (stable)
    return (len(T), tuple(sorted(X)))

def show_top(results, k=10):
    print(f"\nTop {k} closed itemsets (by support desc, then length desc):")
    for X, sup in sorted(results, key=lambda p: (-p[1], -len(p[0]), tuple(sorted(p[0]))))[:k]:
        print(f"{tuple(sorted(X))}: sup={sup}")

### Build vertical format and L1 ####
def build_vertical(transactions, minsup):
    item2tids = defaultdict(set)
    for tid, items in transactions:
        for i in items:
            item2tids[i].add(tid)
    L1 = []
    for i, tids in item2tids.items():
        if len(tids) >= minsup:
            L1.append((frozenset([i]), set(tids)))  # (itemset, tidset)
    L1.sort(key=sort_key)
    return L1

#### Global structures for closedness via tidset identity ####
class ClosedStore:
    """
    Keeps one representative itemset per distinct tidset (closure).
    If two itemsets share the same tidset, we merge (union of items).
    """
    def __init__(self):
        self.tid2items = dict()   # key: frozenset(tids) -> frozenset(items)

    def add_closed(self, X, T):
        key = frozenset(T)
        if key in self.tid2items:
            self.tid2items[key] = self.tid2items[key] | X  # merge (closure)
        else:
            self.tid2items[key] = X

####  CHARM recursion ########
def charm(class_list, minsup, store):
    """
    Mine one equivalence class (all pairs share a common prefix).
    class_list: list of pairs (X, T) with T as a Python set of tids
    """
    m = len(class_list)
    i = 0
    while i < m:
        Xi, Ti = class_list[i]
        new_class = []

        j = i + 1
        while j < m:
            Xj, Tj = class_list[j]
            Tij = Ti & Tj
            if len(Tij) < minsup:
                j += 1
                continue

            Y = Xi | Xj

            if Tij == Ti and Tij == Tj:
                # (P1) same tidset: merge Xj into Xi, absorb Xj
                Xi = Y
                Ti = Tij
                class_list.pop(j)
                m -= 1
                continue

            elif Tij == Ti:
                # (P2) Ti ⊆ Tj: extend Xi in place
                Xi = Y
                # Ti remains Tij (same)
                j += 1
                continue

            elif Tij == Tj:
                # (P3) Tj ⊆ Ti: prune Xj
                class_list.pop(j)
                m -= 1
                continue

            else:
                # (P4) general case: create child candidate under Xi
                new_class.append((Y, Tij))
                j += 1

        # Recurse on children
        if new_class:
            new_class.sort(key=sort_key)
            charm(new_class, minsup, store)

        # After exploring, Xi is a (prefix) closed candidate; store by tidset
        store.add_closed(Xi, Ti)
        i += 1

#####  Driver #######
def run_charm(transactions, minsup):
    L1 = build_vertical(transactions, minsup)
    store = ClosedStore()
    charm(L1, minsup, store)

    # normalize: each entry is (items, support)
    closed = [(items, len(tids)) for tids, items in store.tid2items.items()]
    return closed

# Build vertical map (item -> TID set) once, to evaluate supports of subsets quickly
def build_item2tids(transactions):
    m = {}
    for tid, items in transactions:
        for i in items:
            m.setdefault(i, set()).add(tid)
    return m

def support_count(itemset, item2tids):
    """Absolute support via TID-set intersections (0 if any item is unseen)."""
    if not itemset:
        return 0
    try:
        tidsets = [item2tids[i] for i in itemset]
    except KeyError:
        return 0
    return len(reduce(operator.and_, tidsets))

def rules_from_closed(closed_itemsets, item2tids, N, min_conf=0.6, min_lift=1.0):
    """
    Generate rules X -> Y where X ⊂ Z, Y = Z\X, for each closed itemset Z.
    confidence = sup(Z)/sup(X); lift = confidence / P(Y).
    """
    out = []
    for Z, supZ in closed_itemsets:
        Z = set(Z)
        if len(Z) < 2:
            continue
        for r in range(1, len(Z)):
            for X in combinations(Z, r):
                X = set(X)
                Y = tuple(sorted(Z - X))
                if not Y:
                    continue
                supX = support_count(X, item2tids)
                if supX == 0:
                    continue
                conf = supZ / supX
                supY = support_count(Y, item2tids)
                lift = conf / (supY / N) if supY > 0 else float("inf")
                if conf >= min_conf and lift >= min_lift:
                    out.append({
                        "antecedent": tuple(sorted(X)),
                        "consequent": Y,
                        "support": supZ,
                        "support_frac": supZ / N,
                        "confidence": conf,
                        "lift": lift,
                        "rule_len": len(Z),
                    })
    cols = ["antecedent","consequent","support","support_frac","confidence","lift","rule_len"]
    if not out:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(out)
              .sort_values(["lift","confidence","support","rule_len"],
                           ascending=[False,False,False,False])
              .reset_index(drop=True))

def experiment(transactions, minsup_fracs):
    rows = []
    for ms in minsup_fracs:
        abs_ms = ceil(ms * len(transactions))
        tracemalloc.start()
        t0 = time.perf_counter()
        closed = run_charm(transactions, abs_ms)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({
            "minsup_frac": ms,
            "minsup_abs": abs_ms,
            "#closed_itemsets": len(closed),
            "runtime_sec": t1 - t0,
            "peak_memory_MB": peak / (1024**2)
        })
    return pd.DataFrame(rows)

#################################################### Main Program ##########################################################

def main():
    groceries_data_original = pd.read_csv("Groceries_dataset.csv")
    groceries_data = groceries_data_original.copy()
    #drop time, keep date for grouping
    groceries_data['Date'] = pd.to_datetime(groceries_data['Date'], format="mixed", dayfirst=True)
    groceries_data['Date'] = groceries_data['Date'].dt.date
    # print(groceries_data.info())
    # print(groceries_data['Date'].value_counts())

    # Aggregate to one basket per member and date (assuming 1 transaction / day per person)
    item_List_df = (groceries_data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)).reset_index(name='items')
    # print(item_List_df.head())

    #### Build transactions (one TID per (Member_number, Date)) ####
    transactions = [(tid, set(row.items))
                    for tid, row in enumerate(item_List_df.itertuples(index=False))]
    n_transactions = len(transactions)
    print(f"# of transactions: {n_transactions}")

    # -------- EDIT HERE: initial mining minsup --------
    minsup_frac = 0.01             # <-- fraction of transactions (e.g., 1%)
    minsup_abs = ceil(minsup_frac * n_transactions)
    # -----------------------------------------------

    print(f"minsup (fraction, abs): {minsup_frac} -> {minsup_abs}")

    # Warm-up (optional; avoids first-run overheads biasing timings)
    _ = run_charm(transactions, max(2, ceil(0.25 * n_transactions)))

    # Mine closed frequent itemsets
    closed_itemsets = run_charm(transactions, minsup_abs)
    print(f"\n#closed frequent itemsets: {len(closed_itemsets)}")
    show_top(closed_itemsets, k=15)

    # Save closed itemsets
    closed_df = (pd.DataFrame(
                    [{"itemset": tuple(sorted(items)),
                      "length": len(items),
                      "support": sup}
                     for items, sup in closed_itemsets])
                 .sort_values(["support","length","itemset"],
                              ascending=[False, False, True])
                 .reset_index(drop=True))
    print("\nClosed itemsets (head):")
    print(closed_df.head(10))
    closed_df.to_csv("closed_itemsets.csv", index=False)
    print("Saved closed itemsets -> closed_itemsets.csv")

    # Rule mining from closed itemsets
    item2tids = build_item2tids(transactions)

    # -------- EDIT HERE: rule thresholds --------
    MIN_CONF = 0.60   # confidence threshold
    MIN_LIFT = 1.10   # lift threshold
    # -------------------------------------------

    rules_df = rules_from_closed(closed_itemsets, item2tids, n_transactions,
                                 min_conf=MIN_CONF, min_lift=MIN_LIFT)
    print(f"\n#rules (min_conf={MIN_CONF}, min_lift={MIN_LIFT}): {len(rules_df)}")
    print(rules_df.head(15).to_string(index=False))
    rules_df.head(50).to_csv("representative_rules.csv", index=False)
    print("Saved representative rules -> representative_rules.csv")

    # -------- EDIT HERE: minsup sweep for experiments --------
    minsups = [0.20, 0.10, 0.05, 0.02, 0.01]  # try adding 0.005 if runtime allows
    # ---------------------------------------------------------

    summary_df = experiment(transactions, minsups)
    print("\nSweep summary:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("charm_minsup_sweep.csv", index=False)
    print("Saved sweep table -> charm_minsup_sweep.csv")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(summary_df["minsup_frac"], summary_df["runtime_sec"], marker="o")
    ax[0].set_xlabel("minsup (fraction)"); ax[0].set_ylabel("runtime (s)")
    ax[0].set_title("Runtime vs minsup"); ax[0].invert_xaxis()

    ax[1].plot(summary_df["minsup_frac"], summary_df["#closed_itemsets"], marker="o")
    ax[1].set_xlabel("minsup (fraction)"); ax[1].set_ylabel("# closed itemsets")
    ax[1].set_title("Count vs minsup"); ax[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig("charm_runtime_count_vs_minsup.png", dpi=200, bbox_inches="tight")
    print("Saved plot -> charm_runtime_count_vs_minsup.png")

if __name__ == "__main__":
    main()