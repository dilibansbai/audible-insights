import pandas as pd

def recommend_content(df, cosine_sim, title, k=5):
    title = title.lower()
    
    matches = df[df['title'].str.contains(title)]
    if matches.empty:
        return pd.DataFrame()
    
    idx = matches.index[0]

    sim_scores = cosine_sim[idx].flatten()
    
    scores = list(enumerate(sim_scores))
    
    scores = sorted(scores, key=lambda x: float(x[1]), reverse=True)[1:k+1]
    
    indices = [i[0] for i in scores]
    
    return df[['title', 'author', 'rating']].iloc[indices]

def recommend_cluster(df,title, k=5):
    title = title.lower()
    matches = df[df['title'].str.contains(title)]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    cluster_id = df.loc[idx, 'cluster']
    
    recs = df[df['cluster'] == cluster_id]
    recs = recs[recs.index != idx]  # remove same book
    
    return recs[['title','author','rating']].head(k).reset_index(drop=True)

def recommend_hybrid(df, cosine_sim, title, k=5):

    content = recommend_content(df, cosine_sim, title, k)
    cluster = recommend_cluster(df, title, k)

    hybrid = content.head(3).copy()

    for _, row in cluster.iterrows():
        if row["title"] not in hybrid["title"].values:
            hybrid = pd.concat(
                [hybrid, row.to_frame().T],
                ignore_index=True
            )

        if len(hybrid) >= k:
            break

    return hybrid.reset_index(drop=True)

def precision_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    relevant = set(relevant)
    
    hits = len([r for r in recommended if r in relevant])
    return hits / k

def recall_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    relevant = set(relevant)
    
    hits = len([r for r in recommended if r in relevant])
    return hits / len(relevant) if len(relevant) > 0 else 0