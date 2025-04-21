import pandas as pd
from collections import Counter

def update_history(history_list, query):
    history_list.append(query)

def get_history_df(history_list):
    if not history_list:
        return pd.DataFrame(columns=["Query", "Frequency"])

    freq = Counter(history_list)
    data = [{"Query": q, "Frequency": f} for q, f in freq.items()]
    return pd.DataFrame(data).sort_values(by="Frequency", ascending=False)
