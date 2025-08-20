import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("world_population.csv")

# Convert population columns to numeric
pop_cols = ['1970','1980','1990',
            '2000','2010','2015',
            '2020','2022']
for col in pop_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------------------------------
# 1. World Population Growth (1970–2022)
# ----------------------------------------------------
world_pop = df[pop_cols].sum()
plt.figure(figsize=(10,6))
world_pop.plot(marker='o')
plt.title("Worl Population Growth (1970–2022)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.grid(True)
plt.show()

# ----------------------------------------------------
# 2. Top 10 Countries by Population (2022)
# ----------------------------------------------------
top10 = df[['Country/Territory','2022']].nlargest(10, '2022')
plt.figure(figsize=(10,6))
sns.barplot(x='2022', y='Country/Territory', data=top10, palette="viridis")
plt.title("Top 10 Most Populous Countries (2022)")
plt.xlabel("Population")
plt.ylabel("Country")
plt.show()

# ----------------------------------------------------
# 3. Population Share by Continents (2022)
# ----------------------------------------------------
continent_share = df.groupby("Continent")["2022"].sum()
plt.figure(figsize=(8,8))
plt.pie(continent_share, labels=continent_share.index, autopct='%1.1f%%', startangle=140)
plt.title("Population Share by Continent (2022)")
plt.show()

# ----------------------------------------------------
# 4. Population Growth: India vs China (1970–2022)
# ----------------------------------------------------
india = df[df['Country/Territory'] == "India"][pop_cols].values.flatten()
china = df[df['Country/Territory'] == "China"][pop_cols].values.flatten()

years = [int(c.split()[0]) for c in pop_cols]
plt.figure(figsize=(10,6))
plt.plot(years, india, marker='o', label="India")
plt.plot(years, china, marker='o', label="China")
plt.title("Population Growth: India vs China")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------
# 5. Top 15 Countries by Absolute Growth since 2000
# ----------------------------------------------------
df['Growth_2000_2022'] = df['2022'] - df['2000']
top15_growth = df.nlargest(15, 'Growth_2000_2022')

plt.figure(figsize=(10,6))
sns.barplot(x='Growth_2000_2022', y='Country/Territory', data=top15_growth, palette="mako")
plt.title("Top 15 Countries by Absolute Population Growth (2000–2022)")
plt.xlabel("Growth in Population")
plt.ylabel("Country")
plt.show()

# ----------------------------------------------------
# 6. Decadal CAGR Heatmap
# ----------------------------------------------------
def calc_cagr(start, end, years):
    return (end/start)**(1/years) - 1 if start>0 else None

cagr_data = []
for _, row in df.iterrows():
    country = row['Country/Territory']
    for i in range(len(pop_cols)-1):
        start_year = int(pop_cols[i].split()[0])
        end_year = int(pop_cols[i+1].split()[0])
        start_val = row[pop_cols[i]]
        end_val = row[pop_cols[i+1]]
        cagr = calc_cagr(start_val, end_val, end_year-start_year)
        cagr_data.append([country, f"{start_year}-{end_year}", cagr])

cagr_df = pd.DataFrame(cagr_data, columns=['Country','Period','CAGR'])


# ---- Heatmap of CAGR by Country and Period ----
# ---- Calculate CAGR for each country (2000–2020) ----
def cagr(start_value, end_value, years):
    return (end_value / start_value) ** (1/years) - 1 if start_value > 0 else np.nan

cagr_data = []
for _, row in df.iterrows():
    country = row["Country"]
    try:
        pop_2000 = row["2000"]
        pop_2020 = row["2020"]
        if not pd.isna(pop_2000) and not pd.isna(pop_2020) and pop_2000 > 0:
            growth = cagr(pop_2000, pop_2020, 20) * 100
            cagr_data.append({"Country": country, "CAGR (%)": growth, "Continent": row["Continent"]})
    except:
        continue

cagr_df = pd.DataFrame(cagr_data)

# ---- Top 10 countries by CAGR ----
top10_cagr = cagr_df.sort_values(by="CAGR (%)", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top10_cagr, x="CAGR (%)", y="Country", palette="viridis")
plt.title("Top 10 Countries by Population CAGR (2000–2020)")
plt.xlabel("CAGR (%)")
plt.ylabel("Country")
plt.show()

# ---- Regional CAGR ----
regional_cagr = cagr_df.groupby("Continent")["CAGR (%)"].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8,6))
sns.barplot(data=regional_cagr, x="CAGR (%)", y="Continent", palette="magma")
plt.title("Average Population CAGR by Region (2000–2020)")
plt.xlabel("CAGR (%)")
plt.ylabel("Continent")
plt.show()

