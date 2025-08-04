import requests
from bs4 import BeautifulSoup
import csv

urls = [
    "https://beminimalist.co/",
    "https://www.geeksforgeeks.org/",
    "https://www.morphe.com/"
]

csv_rows = []

for url in urls:
    print(f"\n Headlines from: {url}")
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    headlines = []
    for tag in ['h1', 'h2', 'h3']:
        for element in soup.find_all(tag):
            text = element.get_text(strip=True)
            if text:
                headlines.append(text)

    top_10 = headlines[:10]

    for i, headline in enumerate(top_10, start=1):
        if i == 1:
            csv_rows.append([url, i, headline])  # First row with website
        else:
            csv_rows.append(["", i, headline])   # Subsequent rows with blank website
        print(f"{i}. {headline}")

with open("headlines.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Website", "No.", "Headline"])
    writer.writerows(csv_rows)

print("\n Saved numbered headlines to headlines.csv")