import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
num_samples = 1000
molecular_weights = np.random.uniform(200, 600, num_samples)
solubilities = np.random.uniform(0.1, 1.0, num_samples)
hydrophobicities = np.random.uniform(0.1, 1.0, num_samples)
logP_values = np.random.uniform(0.1, 5.0, num_samples)
toxicities = np.random.randint(0, 2, num_samples)

data = pd.DataFrame({
    'Molecular_Weight': molecular_weights,
    'Solubility': solubilities,
    'Hydrophobicity': hydrophobicities,
    'LogP': logP_values,
    'Toxicity': toxicities
})

# Save dataset to CSV
data.to_csv('synthetic_drug_data.csv', index=False)
print("Synthetic dataset saved as 'synthetic_drug_data.csv'")
