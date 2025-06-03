import os
import pandas as pd
from typing import List, Tuple


def extract_response_times(directory: str) -> List[Tuple[str, float, float, float]]:
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and "csv_stats." in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            aggregated = df[df["Name"] == "Aggregated"]
            print(aggregated)
            if not aggregated.empty:
                median = aggregated["Median Response Time"].iloc[0]
                p95 = aggregated["95%"].iloc[0]
                p98 = aggregated["98%"].iloc[0]
                results.append((filename.split(".")[0], median, p95, p98))

    df = pd.DataFrame(
        results, columns=["Approach", "Median Response Time", "95%", "98%"]
    )
    markdown_table = df.to_markdown(index=False)
    return df, markdown_table
