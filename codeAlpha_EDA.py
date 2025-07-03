# TASK 2: Exploratory Data Analysis (EDA) on Job Listings Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv("unbelievable_data_analyst_jobs.csv")

# ------------------- Step 1: Ask Questions ------------------- #
print("❓ Key Questions to Explore:\n")
print("""
1. Which companies are hiring the most for Data Analyst roles?
2. Which locations have the highest number of openings?
3. Are there duplicate or missing values?
4. What are the most common keywords in job descriptions?
5. Is there a correlation between companies and locations?
""")

# ------------------- Step 2: Data Structure ------------------- #
print("\n🧱 Dataset Structure:\n")
print(df.info())

# ------------------- Step 3: Preview the Data ------------------- #
print("\n🔍 Preview:\n", df.head())

# ------------------- Step 4: Missing & Duplicate Values ------------------- #
print("\n🚨 Missing Values:\n", df.isnull().sum())
print("\n🧹 Duplicate Entries:", df.duplicated().sum())

# ------------------- Step 5: Unique Counts ------------------- #
print("\n🔢 Unique Titles:", df['Title'].nunique())
print("🔢 Unique Companies:", df['Company'].nunique())
print("🔢 Unique Locations:", df['Location'].nunique())

# ------------------- Step 6: Top Companies ------------------- #
top_companies = df['Company'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_companies.values, y=top_companies.index, palette='Blues_r')
plt.title("🏢 Top Hiring Companies")
plt.xlabel("Number of Job Listings")
plt.ylabel("Company")
plt.show()

# ------------------- Step 7: Top Locations ------------------- #
top_locations = df['Location'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_locations.values, y=top_locations.index, palette='magma')
plt.title("📍 Top Locations for Data Analyst Jobs")
plt.xlabel("Number of Listings")
plt.ylabel("Location")
plt.show()

# ------------------- Step 8: Word Frequency in Summary ------------------- #
from collections import Counter
import re

all_words = ' '.join(df['Summary']).lower()
tokens = re.findall(r'\b\w+\b', all_words)
common_words = Counter(tokens).most_common(20)

words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
plt.figure(figsize=(10,6))
sns.barplot(x='Frequency', y='Word', data=words_df, palette='coolwarm')
plt.title("🔤 Most Frequent Words in Job Descriptions")
plt.show()

# ------------------- Step 9: Hypothesis Testing ------------------- #
# Is job location dependent on company?
contingency_table = pd.crosstab(df['Company'], df['Location'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\n📊 Chi-square Test: p-value = {p:.5f}")
if p < 0.05:
    print("✅ Statistically significant: Job location depends on company.")
else:
    print("❌ No significant relationship between company and job location.")

# ------------------- Step 10: Data Quality Checks ------------------- #
print("\n🔎 Checking for inconsistencies...")
df['Location'] = df['Location'].str.strip()
df['Company'] = df['Company'].str.strip()

print("✅ Cleaned whitespace from 'Location' and 'Company'.")
