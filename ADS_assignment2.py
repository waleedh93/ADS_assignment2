#importing libraries in python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial as Poly
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

# importing the csv file for analysis
df = pd.read_csv('crop_production.csv')
print(df.head())

print(df.describe(include = 'all'))
new_df = df.drop(['INDICATOR','FREQUENCY','Flag Codes'],axis = 1)
#print(new_df)

countries = ['AUS', 'USA', 'CAN']
filtered_df = new_df[(new_df['LOCATION'].isin(countries))&(new_df['SUBJECT'].str.contains('WHEAT', na=False))  & (new_df['MEASURE']== 'TONNE_HA')]
print(filtered_df)
#print(df['SUBJECT'].dtype)
#crop_types = filtered_df['SUBJECT'].unique()  # Get unique crop types
#print(crop_types)
# Loop through each crop type and plot the yield for each
for country in countries:
    country_data = filtered_df[filtered_df['LOCATION'] == country]  # Filter data for the current crop
    Time = country_data.iloc[:, -2]  # Second-to-last column (Time)
    Value = country_data.iloc[:, -1]  # Last column (Value - yield)
    plt.plot(Time, Value)
plt.legend(['AUS', 'USA', 'CAN'])
plt.xlabel('Years Since 1990')
plt.ylabel('Yield (TONNE_HA)')
plt.title('Wheat crop yield of three countries')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

# Pivot the data for the heatmap
heatmap_data = filtered_df.pivot_table( index='TIME',columns='LOCATION', values='Value')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data,  annot=True)
plt.title('Heatmap of Wheat Yield (TONNE_HA)')
plt.xlabel('Country')
plt.ylabel('Year')
plt.show()


# Pivot the data to align years and countries as columns
pivoted_data = filtered_df.pivot_table(index='TIME', columns='LOCATION', values='Value')

# Compute correlation matrix
correlation_matrix = pivoted_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation of Wheat Yield Across Countries')
plt.show()


# plotting the box and violin plot

sns.boxplot(x='LOCATION', y='Value', data=filtered_df)
plt.title("Box Plot of Wheat Yield")
plt.ylabel("Wheat Yield")
plt.tight_layout()
plt.show()


#plotting the violin plot

sns.violinplot(x='LOCATION', y='Value', data=filtered_df)
plt.title("Violin Plot of Wheat Yield")
plt.ylabel("Yield Value")
plt.tight_layout()  # Adjust layout for rotated labels and landscape orientation
plt.show()

## fitting the curve
filtered_dff = new_df[(new_df['LOCATION'].str.contains('USA', na=False))&(new_df['SUBJECT'].str.contains('WHEAT', na=False))& (new_df['MEASURE'] == 'TONNE_HA')]
print(filtered_dff)
Time = filtered_dff['TIME']  # Assuming 'TIME' is the column name for years
Value = filtered_dff['Value']  # Assuming 'Value' is the column name for yield
plt.scatter(Time, Value)
plt.legend(['USA'])
plt.xlabel('Years')
plt.ylabel('Yield (TONNE_HA)')
plt.title('Wheat crop yield of the USA')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()
def exponential(t, n0, g):
    f = n0 * np.exp(g*t)
    return f
numeric_index = (Time - 1990).values
p0 = (2.25, 0.02)
p, cov = curve_fit(exponential, numeric_index, Value, p0=p0, maxfev=10000)
sigma = np.sqrt(np.diag(cov))
print(f"N0 = {p[0]:g} +/- {sigma[0]:g}")
print(f"g = {p[1]:.2f} +/- {sigma[1]:.2f}")
#print(f"t0 = {p[2] + 1990:.2f} +/- {sigma[2]:.2f}")
fig, ax = plt.subplots(dpi=144)
filtered_dff['Exponential Fit'] = exponential(numeric_index, *p)
filtered_dff.plot(y=['Value', 'Exponential Fit'],  ax=ax, ylabel='Value')
plt.show()

#numeric_index = (filtered_dff.numeric_index - 1990).values
p0 = (2.25, 0.02)
p, cov = curve_fit(exponential, numeric_index, filtered_dff['Value'],
p0=p0)
Value_2030 = exponential(2030 - 1990, *p) # remember to subtract the 1990 as we did when 'training'
print(f"Value in 2030: {Value_2030:g}")

# take 1000 normal random samples for each parameter
sample_params = ss.multivariate_normal.rvs(mean=p, cov=cov, size=1000)
# standard deviation of all possible parameter sampling
Value_unc_2030 = np.std(exponential(2030 - 1990, *sample_params.T)) # note the transpose
print(f"Value in 2030: {Value_2030:g} +/- {Value_unc_2030:g}")
fig, ax = plt.subplots(dpi=144)
# create array of values within data, and beyond
filtered_dff.sort_values(by='TIME', inplace=True)
time_predictions = np.arange(1990, 2035, 5)
# determine predictions for each of those times
value_predictions = exponential(time_predictions - 1990, *p)
# determine uncertainty at each prediction
value_uncertainties = [np.std(exponential(future_time - 1990, *sample_params.T)) for future_time in time_predictions]
ax.plot(filtered_dff['TIME'], filtered_dff['Value'], 'b-', label='Data')
ax.plot(time_predictions, value_predictions, 'k-', label='Exponential Fit')
ax.fill_between(time_predictions, value_predictions - value_uncertainties,value_predictions + value_uncertainties,color='gray', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Yield Value')
ax.legend()

#ax.set_yscale('log')
plt.grid('True')
plt.show()

### clustering

# Convert numeric columns explicitly
for col in filtered_df.columns:
    if filtered_df[col].dtype == 'object':
        continue  # Skip non-numeric columns
    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

# Drop rows with NaN values (optional: check for columns with many NaNs)
filtered_df = filtered_df.dropna()

# Debugging: Print a summary of the dataset
print("Summary after cleaning:")
print(filtered_df.info())

# Pairplot visualization
sns.pairplot(filtered_df, corner=True)
plt.show()

# Generate heatmap
fig, ax = plt.subplots(dpi=144)
mask = np.triu(np.ones_like(filtered_df.corr(numeric_only=True), dtype=bool))
sns.heatmap(filtered_df.corr(numeric_only=True), ax=ax, vmin=-1, vmax=1, cmap='RdBu', annot=True, mask=mask)
plt.show()
def plot_elbow_method(min_k, max_k, wcss, best_n):
    fig, ax = plt.subplots(dpi=144)
    ax.plot(range(min_k, max_k + 1), wcss, 'kx-')
    ax.scatter(best_n, wcss[best_n-min_k], marker='o', color='red',facecolors='none', s=50)
    ax.set_xlabel('k')
    ax.set_xlim(min_k, max_k)
    ax.set_ylabel('WCSS')
    plt.show()
    return
from sklearn.preprocessing import RobustScaler
df_clust = filtered_df[['TIME', 'Value']].copy()
scaler = RobustScaler()
norm = scaler.fit_transform(df_clust)
def one_silhoutte_inertia(n, xy):
    kmeans = KMeans(n_clusters=n, n_init=20)
    # Fit the data
    kmeans.fit(xy)
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = silhouette_score(xy, labels)
    inertia = kmeans.inertia_
    return score, inertia
wcss = []
best_n, best_score = None, -np.inf
for n in range(2, 11): # 2 to 10 clusters
      score, inertia = one_silhoutte_inertia(n, norm)
      wcss.append(inertia)
      if score > best_score:
          best_n = n
          best_score = score
print(f"{n:2g} clusters silhoutte score = {score:0.2f}")
print(f"Best number of clusters = {best_n:2g}")
plot_elbow_method(2, 10, wcss, best_n)
def plot_value_time(labels, xy, xkmeans, ykmeans, centre_labels):
    colours = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
    cmap = ListedColormap(colours)
    fig, ax = plt.subplots(dpi=144)
    s = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, marker='o',label='Data')
    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=cmap, marker='x', s=100,label='Estimated Centres')
    cbar = fig.colorbar(s, ax=ax)
    cbar.set_ticks(np.unique(labels))
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
   # ax.set_xscale('log')
    plt.show()
    return
inv_norm = scaler.inverse_transform(norm) # this is important for plotting data accurately
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, n_init=20)
    kmeans.fit(norm) # fit done on x,y pairs
    labels = kmeans.labels_
    # the estimated cluster centres
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]
    cenlabels = kmeans.predict(kmeans.cluster_centers_)
    plot_value_time(labels, inv_norm, xkmeans, ykmeans, cenlabels)
