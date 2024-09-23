
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Data Preparation
data = {
    'Brand': ['ALFA ROMEO', 'AUDI', 'BENTLEY', 'BMW', 'CITROEN', 'DACIA', 'DS', 'FERRARI', 'FIAT', 'FORD', 
              'HONDA', 'HYUNDAI', 'JAGUAR', 'JEEP', 'KIA', 'LADA', 'LAMBORGHINI', 'LAND ROVER', 'LEVC', 
              'LEXUS', 'LYNK & CO', 'MAN', 'MASERATI', 'MAZDA', 'MERCEDES', 'MG ROEWE', 'MINI', 'MITSUBISHI',
              'NISSAN', 'OPEL', 'PEUGEOT', 'POLESTAR', 'PORSCHE', 'RENAULT', 'SEAT', 'SKODA', 'SMART', 
              'SSANGYONG', 'SUBARU', 'SUZUKI', 'TESLA', 'TOYOTA', 'VOLVO', 'VW'],
    
    'New Registrations Q1 2022': [631, 53036, 250, 50245, 8338, 12560, 541, 389, 17970, 30013, 2004, 
                                  23431, 861, 3613, 16306, 459, 261, 2120, 2, 787, 467, 265, 209, 
                                  9549, 57602, 1371, 11036, 8465, 6096, 34681, 11111, 1109, 7396, 
                                  19735, 29383, 37206, 4299, 589, 1291, 3151, 14408, 17997, 9147, 
                                  112025],
    
    'Alternative Drive Registrations Q1 2022': [2, 37608, 22, 33245, 1383, 5998, 270, 52, 9365, 13842, 
                                                1810, 15165, 427, 2001, 7481, 4, 2, 1762, 2, 728, 467, 
                                                3, 67, 5354, 31284, 1371, 3189, 3110, 3849, 5463, 5255, 
                                                1109, 1708, 9996, 7023, 5423, 4299, 92, 640, 3140, 14408, 
                                                14072, 8231, 16798],
    
    'Percentage with Alternative Drive Q1 2022': [0.3, 70.9, 8.8, 66.2, 16.6, 47.8, 49.9, 13.4, 52.1, 46.1,
                                                  90.3, 64.7, 49.6, 55.4, 45.9, 0.9, 0.8, 83.1, 100, 92.5,
                                                  100, 1.1, 32.1, 56.1, 54.3, 100, 28.9, 36.7, 63.1, 15.8, 
                                                  47.3, 100, 23.1, 50.7, 23.9, 14.6, 100, 15.6, 49.6, 99.7, 
                                                  100, 78.2, 90, 15]
}

df = pd.DataFrame(data)

# Step 2: Additional Plots

# Plot 1: Bar Plot of Top 10 Brands by Total New Registrations
plt.figure(figsize=(10,6))
top_10_brands = df.nlargest(10, 'New Registrations Q1 2022')
sns.barplot(x='New Registrations Q1 2022', y='Brand', data=top_10_brands, palette='Blues_d')
plt.title('Top 10 Brands by New Registrations in Q1 2022')
plt.xlabel('Number of New Registrations')
plt.ylabel('Brand')
plt.show()

# Plot 2: Distribution of Percentage with Alternative Drive (All Brands)
plt.figure(figsize=(10,6))
sns.histplot(df['Percentage with Alternative Drive Q1 2022'], kde=True, bins=10, color='purple')
plt.title('Distribution of Alternative Drive Percentages Across All Brands')
plt.xlabel('Percentage with Alternative Drive Q1 2022')
plt.ylabel('Frequency')
plt.show()

# Plot 3: Correlation Heatmap for New Registrations and Alternative Drive Percentage
plt.figure(figsize=(8,6))
correlation_matrix = df[['New Registrations Q1 2022', 'Alternative Drive Registrations Q1 2022', 'Percentage with Alternative Drive Q1 2022']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: New Registrations and Alternative Drive Data')
plt.show()

# Plot 4: Line Plot for EV Adoption by Selected Brands Over Time (Hypothetical Data)
# Creating hypothetical time-series data for EV adoption trends for selected brands
time_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Audi': [500, 520, 550, 600, 620, 630, 640, 700, 710, 750, 770, 800],
    'BMW': [450, 460, 470, 500, 520, 530, 540, 590, 600, 620, 630, 650],
    'Mercedes': [400, 410, 420, 450, 470, 480, 490, 540, 550, 570, 590, 620]
})

plt.figure(figsize=(10,6))
sns.lineplot(x='Month', y='Audi', data=time_data, label='Audi')
sns.lineplot(x='Month', y='BMW', data=time_data, label='BMW')
sns.lineplot(x='Month', y='Mercedes', data=time_data, label='Mercedes')
plt.title('EV Adoption Trends Over Time for Selected Brands (Hypothetical)')
plt.xlabel('Month')
plt.ylabel('EV Registrations')
plt.legend()
plt.show()
