#clickstream data analysis
#understand how user navigaate through their website
#session paths - sequence of pages visited during a single session
#funnels - how many users reahc each step of a purchase process and where drop offsz happen 
#navigataion visualisation - most common paths users take 
#simulate clickstream data, analyze session behaviour, identify bottlenecks and visualize navigation flowsimport pandas as pd
import pandas as pd
import random
from datetime import datetime, timedelta
from collections import Counter

#1: Simulate Clickstream Data
users = [f"user_{i}" for i in range(1, 21)]
pages = ["Home", "Search", "Product", "Cart", "Checkout", "Confirmation"]

clickstream = []

for user in users:
    session_length = random.randint(2, 6)
    start_time = datetime(2025, 10, 8, 9, 0)
    for i in range(session_length):
        page = pages[i] if i < len(pages) else random.choice(pages)
        timestamp = start_time + timedelta(minutes=random.randint(1, 5) * i)
        clickstream.append({
            "User": user,
            "Page": page,
            "Timestamp": timestamp
        })

#2: Create DataFrame and sort
df = pd.DataFrame(clickstream)
df.sort_values(by=["User", "Timestamp"], inplace=True)

#3: Display Simulated Clickstream Data
print("\nSimulated Clickstream Data:")
print(f"{'User':<10} {'Page':<12} {'Timestamp'}")
print("-" * 40)
for row in df.itertuples(index=False):
    print(f"{row.User:<10} {row.Page:<12} {row.Timestamp}")

#4: Analyze Session Paths
session_paths = df.groupby("User")["Page"].apply(list)

print("\nSession Paths:")
for user, path in session_paths.items():
    print(f"{user}: {' -> '.join(path)}")

#5: Funnel Analysis
funnel_steps = ["Home", "Search", "Product", "Cart", "Checkout", "Confirmation"]
funnel_counts = {step: 0 for step in funnel_steps}

for path in session_paths:
    for step in funnel_steps:
        if step in path:
            funnel_counts[step] += 1

print("\nFunnel Step Counts:")
for step, count in funnel_counts.items():
    print(f"{step}: {count} users")

#6: Navigation Flow Analysis
transitions = []
for path in session_paths:
    for i in range(len(path) - 1):
        transitions.append((path[i], path[i + 1]))

transition_counts = Counter(transitions)

print("\nNavigation Flows:")
for (from_page, to_page), count in transition_counts.items():
    print(f"{from_page} -> {to_page}: {count} times")