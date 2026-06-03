import pandas as pd

def rank_by_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    out = df.copy(); out["rank"] = out.groupby("dataset")[metric_col].rank(method="average"); return out
