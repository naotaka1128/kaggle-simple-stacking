import pandas as pd

def probability_to_rank(pred):
    df = pd.DataFrame(columns=['probability'])
    df['probability'] = pred
    df['rank'] = df['probability'].rank() / len(pred)
    return df['rank'].values
