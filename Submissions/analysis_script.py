
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Step 2: Hypothesis 1 - Premium vs Non-Premium Brands for Alternative Drives

plt.figure(figsize=(10,6))
premium_brands = ['Audi', 'BMW', 'Mercedes']
non_premium_brands = ['Fiat', 'Dacia', 'Ford']

df['Brand Type'] = df['Brand'].apply(lambda x: 'Premium' if x in premium_brands else 'Non-Premium')

sns.boxplot(x='Brand Type', y='Percentage with Alternative Drive Q1 2022', data=df, palette='Set2')
plt.title('Percentage of Alternative Drive Vehicles: Premium vs Non-Premium Brands')
plt.ylabel('Percentage with Alternative Drive')
plt.xlabel('Brand Type')
plt.show()

# Step 3: Hypothesis 3 - Scatter Plot for New Registrations vs Alternative Drives

plt.figure(figsize=(10,6))
sns.scatterplot(x='New Registrations Q1 2022', y='Percentage with Alternative Drive Q1 2022', hue='Brand', data=df, palette='coolwarm', s=100)
plt.title('New Registrations vs Percentage of Alternative Drive Vehicles by Brand')
plt.xlabel('New Registrations Q1 2022')
plt.ylabel('Percentage with Alternative Drive Q1 2022')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Step 4: Hypothesis 2 - Urban vs Rural Adoption (Hypothetical Data)

urban_rural_data = pd.DataFrame({
    'Region': ['Urban', 'Rural'],
    'Percentage with Alternative Drive': [45.0, 28.5]
})

plt.figure(figsize=(8,6))
sns.barplot(x='Region', y='Percentage with Alternative Drive', data=urban_rural_data, palette='husl')
plt.title('EV Adoption: Urban vs Rural Regions')
plt.ylabel('Percentage with Alternative Drive')
plt.show()

# Step 5: Hypothesis 4 - Incentives vs EV Adoption (Hypothetical Data)

incentive_data = pd.DataFrame({
    'Brand': ['Audi', 'BMW', 'Mercedes', 'Fiat', 'Dacia', 'Ford'],
    'Incentive Amount (Euros)': [4000, 5000, 4500, 2000, 1500, 1800],
    'Percentage with Alternative Drive': [70.9, 66.2, 54.3, 52.1, 47.8, 46.1]
})

plt.figure(figsize=(8,6))
sns.scatterplot(x='Incentive Amount (Euros)', y='Percentage with Alternative Drive', data=incentive_data, hue='Brand', s=150, palette='cool')
plt.title('Incentives vs EV Adoption by Brand')
plt.xlabel('Incentive Amount (Euros)')
plt.ylabel('Percentage with Alternative Drive')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
