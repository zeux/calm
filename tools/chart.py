import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os.path
import time

argp = argparse.ArgumentParser()
argp.add_argument("input", type=str)
argp.add_argument("--output", type=str, default="chart.png")
args = argp.parse_args()

# Function to parse the markdown table and return a DataFrame
def parse_markdown_table(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Finding the start and end of the table
    start, end = None, None
    for i, line in enumerate(lines):
        if line.startswith('|'):
            start = i if start is None else start
            end = i
        elif start is not None:
            break

    # Extracting the table content
    table_lines = lines[start:end+1]

    headers = [header.strip() for header in table_lines[0].split("|")[1:-1]]
    data = []
    for line in table_lines[1:]:
        if "----" in line:
            continue
        row = [value.strip() for value in line.split("|")[1:-1]]
        if len(row) == len(headers):
            data.append(row)

    return pd.DataFrame(data, columns=headers)

# Function to extract numerical values from the table cells
def extract_values(cell):
    tokens = re.search(r"(\d+)\s*tok/s", cell)
    gb_s = re.search(r"(\d+)\s*GB/s", cell)
    return int(tokens.group(1)) if tokens else None, int(gb_s.group(1)) if gb_s else None

# Parsing the Markdown table
df = parse_markdown_table(args.input)

# Extracting and processing the data
df['Tokens/s'], df['GB/s'] = zip(*df['Performance (first 32 tokens)'].apply(extract_values))

# Creating the scatter plot
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
for index, row in df.iterrows():
    plt.scatter(row['Tokens/s'], row['GB/s'], color=colors[index], label=row['Model (context)'], s=100, marker='o', linewidths=2, zorder=3)

mtime = os.path.getmtime(args.input)

plt.xscale('log')

# Customizing the ticks on the Tokens/s axis
# Generating approximately 10 evenly spaced ticks
min_tok_s = min(df['Tokens/s'])
max_tok_s = max(df['Tokens/s'])
ticks = np.logspace(np.log10(min_tok_s), np.log10(max_tok_s), num=10)
plt.xticks(ticks, [f"{int(tick)}" for tick in ticks])

plt.title('calm performance (RTX 4090), ' + time.strftime("%b %Y", time.localtime(mtime)))
plt.xlabel('Performance (Tokens/s)')
plt.ylabel('Performance (GB/s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, zorder=1)
plt.savefig(args.output, format='png', bbox_inches='tight')
