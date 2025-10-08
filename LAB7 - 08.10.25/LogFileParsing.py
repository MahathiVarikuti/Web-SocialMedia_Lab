# Given an Apache web server log file
# Parse the log file and extract:
# 1. Top IP addresses
# 2. Most common referrers
# 3. Average session duration
# Common use cases: traffic analysis, security, marketing
# Libraries: re, pandas, datetime

import re
import pandas as pd
from datetime import datetime
log_data = """
127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 1024 "https://google.com" "Mozilla/5.0"
192.168.0.5 - - [10/Oct/2023:13:58:12 +0000] "GET /about.html HTTP/1.1" 200 2048 "https://facebook.com" "Mozilla/5.0"
127.0.0.1 - - [10/Oct/2023:14:05:45 +0000] "POST /login HTTP/1.1" 302 512 "https://mysite.com" "Mozilla/5.0"
192.168.0.5 - - [10/Oct/2023:14:15:00 +0000] "GET /contact.html HTTP/1.1" 200 1024 "https://linkedin.com" "Mozilla/5.0"
"""
# Parse log lines
entries = []
for line in log_data.strip().splitlines():
    parts = re.findall(r'(\S+) - - \[(.*?)\] "(.*?)" (\d{3}) \d+ "(.*?)"', line)
    if parts:
        ip, raw_time, request, status, referrer = parts[0]
        time_obj = datetime.strptime(raw_time.split()[0], "%d/%b/%Y:%H:%M:%S")
        method, url, _ = request.split()
        entries.append({
            "IP": ip,
            "Time": time_obj,
            "Method": method,
            "URL": url,
            "Status": int(status),
            "Referrer": referrer
        })

# Display parsed entries
print("Parsed Entries:\n", entries)

# Create DataFrame
df = pd.DataFrame(entries)
# 1. Top IP Addresses
top_ips = df["IP"].value_counts().head(10)
print("Top IP Addresses:", top_ips)

# 2. Most Common Referrers
top_referrers = df["Referrer"].value_counts().head(10)
print("\nMost Common Referrers:", top_referrers)
# 3. Session Duration per IP
def session_span(times):
    return (times.max() - times.min()).total_seconds()

durations = df.groupby("IP")["Time"].agg(session_span)
avg_duration = durations.mean()

print("\nSession Duration per IP (seconds):\n", durations)
print(f"\nAverage Session Duration: {avg_duration:.2f} seconds")