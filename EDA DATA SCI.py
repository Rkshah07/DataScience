# ----------------------------------------
# IMPORT LIBRARIES
# ----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.stats.weightstats import ztest

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
file_path = "F:\\Downloads\\apps (2).csv"
df = pd.read_csv(file_path)

# ----------------------------------------
# DATA CLEANING FOR EDA
# ----------------------------------------
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Clean 'Installs' column
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).str.strip()
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Clean 'Price' column
df['Price'] = df['Price'].str.replace('$', '', regex=True).str.strip()
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# ----------------------------------------
# OBJECTIVE 1: Identify the Most Popular Apps Based on User Engagement
# ----------------------------------------
import matplotlib.cm as cm
import numpy as np

# Sort and select top 13 reviewed apps
top_13_reviewed = df.sort_values(by='Reviews', ascending=False).head(13)

# Set up figure
plt.figure(figsize=(12, 7))

# Generate a color palette
colors = sns.color_palette("viridis", len(top_13_reviewed))

# Plot horizontal bar chart
bars = plt.barh(top_13_reviewed['App'], top_13_reviewed['Reviews'], color=colors)

# Add value labels to each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 100000, bar.get_y() + bar.get_height() / 2,
             f'{int(width):,}', va='center', fontsize=9, color='black')

# Style the chart
plt.xlabel('Number of Reviews', fontsize=12)
plt.title('Top 13 Most Reviewed Apps', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
# ----------------------------------------
# OBJECTIVE 2: Discover the Most Competitive App Categories
# ----------------------------------------
category_counts = df['Category'].value_counts().head(10).reset_index()
category_counts.columns = ['Category', 'Count']

plt.figure(figsize=(10, 6))
sns.barplot(data=category_counts, x='Category', y='Count', hue='Category', palette='flare', legend=False)


plt.title('Top 10 Most Competitive App Categories')
plt.xlabel('App Category')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------------------------
# OBJECTIVE 3: Understand Content Suitability Distribution Across Apps
# ----------------------------------------
content_rating_counts = df['Content_Rating'].value_counts().reset_index()
content_rating_counts.columns = ['Content Rating', 'Count']

# Pie Chart
fig_pie = go.Figure(data=[go.Pie(
    labels=content_rating_counts['Content Rating'],
    values=content_rating_counts['Count'],
    textinfo='percent+label',
    marker=dict(colors=sns.color_palette('pastel', len(content_rating_counts)).as_hex())
)])
fig_pie.update_layout(title='Content Rating Distribution (Pie Chart)')
fig_pie.show()

# Donut Chart
fig_donut = go.Figure(data=[go.Pie(
    labels=content_rating_counts['Content Rating'],
    values=content_rating_counts['Count'],
    hole=0.5,
    textinfo='percent+label',
    marker=dict(colors=sns.color_palette('pastel', len(content_rating_counts)).as_hex())
)])
fig_donut.update_layout(title='Content Rating Distribution (Donut Chart)')
fig_donut.show()

# ----------------------------------------
# OBJECTIVE 4: Compare Free vs Paid App Availability Within Top Categories
# ----------------------------------------
df_clean = df.dropna(subset=['Type', 'Category'])
category_type_counts = df_clean.groupby(['Category', 'Type']).size().unstack(fill_value=0)
top_categories = category_type_counts.sum(axis=1).sort_values(ascending=False).head(10)
top_category_counts = category_type_counts.loc[top_categories.index]

top_category_counts.plot(kind='bar', figsize=(12, 7), color=['coral', 'dodgerblue']
)
plt.title('Number of Free vs. Paid Apps per Category (Top 10 Categories)')
plt.xlabel('App Category')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='App Type')
plt.tight_layout()
plt.show()

# ----------------------------------------
# OBJECTIVE 5: Assess Rating Behavior Based on App Type (Free vs Paid)
# ----------------------------------------
df_ztest = df.dropna(subset=['Rating', 'Type'])
free_ratings = df_ztest[df_ztest['Type'] == 'Free']['Rating']
paid_ratings = df_ztest[df_ztest['Type'] == 'Paid']['Rating']

# Z-Test
z_stat, p_value = ztest(free_ratings, paid_ratings)
print("\n--- Z-Test: Are Free Apps Rated Higher than Paid Apps? ---")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Statistically significant difference in ratings (Reject Null Hypothesis)")
else:
    print("Result: No significant difference in ratings (Fail to Reject Null Hypothesis)")



# Boxplot Comparison for Ratings
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_ztest, x='Type', y='Rating', hue='Type', palette='coolwarm', dodge=False)
plt.title('Rating Comparison: Free vs Paid Apps')
plt.xlabel('App Type')
plt.ylabel('Rating')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ----------------------------------------
## Reviews vs Rating Scatter Plot by Category
# ----------------------------------------
df_scatter = df.dropna(subset=['Reviews', 'Rating'])
df_scatter = df_scatter[df_scatter['Reviews'] < 1e6]

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df_scatter,
    x='Reviews',
    y='Rating',
    hue='Category',
    palette='Set2',
    alpha=0.7,
    edgecolor='w',
    s=80
)
plt.title('Scatter Plot: Reviews vs Rating by Category')
plt.xlabel('Number of Reviews')
plt.ylabel('App Rating')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()

# ----------------------------------------
##EDA: Correlation Heatmap
# ----------------------------------------
corr_matrix = df[['Rating', 'Reviews', 'Installs', 'Price']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Between Key Numeric Variables')
plt.tight_layout()
plt.show()


# ----------------------------------------
# OUTLIER DETECTION AND VISUALIZATION
# ----------------------------------------
# ----------------------------------------
# OUTLIER DETECTION AND VISUALIZATION (ENHANCED)
# ----------------------------------------

column = 'Reviews'  # Change to 'Rating', 'Installs', or 'Price' as needed

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
print(f"\n--- Outlier Detection Report for '{column}' ---")
print(f"Total Outliers Detected: {len(outliers)}")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")
print("Top Outliers:")
print(outliers[[column, 'App']].sort_values(by=column, ascending=False).head())

# Summary stats before and after removing outliers
cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
print(f"\n--- Summary Stats for '{column}' ---")
print("With Outliers:")
print(df[column].describe())
print("\nWithout Outliers:")
print(cleaned_df[column].describe())

# ----------------------------------------
# PLOT: Boxplot WITH outliers
# ----------------------------------------
plt.figure(figsize=(12, 5))
sns.boxplot(x=df[column], color='orchid', linewidth=2, fliersize=5)
plt.title(f'Boxplot of {column} with Outliers', fontsize=14, fontweight='bold')
plt.xlabel(column)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------
# PLOT: Boxplot WITHOUT outliers
# ----------------------------------------
plt.figure(figsize=(12, 5))
sns.boxplot(x=cleaned_df[column], color='mediumseagreen', linewidth=2, fliersize=5)
plt.title(f'Boxplot of {column} without Outliers', fontsize=14, fontweight='bold')
plt.xlabel(column)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


