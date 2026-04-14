import matplotlib.pyplot as plt
import os

# Data representing ECMWF's global 5-day forecast skill (approximate historical values)
years = [1990, 2000, 2010, 2020, 2024]
accuracy_scores = [78, 85, 92, 96, 98] # Higher is better

current_dir = os.path.dirname(os.path.abspath(__file__))

plt.figure(figsize=(10, 5))
plt.plot(years, accuracy_scores, marker='D', linestyle='-', color='forestgreen', linewidth=2)

plt.title('Global Meteorological Advancement: Model Skill Increase (1990-2024)', fontsize=12)
plt.ylabel('Model Accuracy Score (%)')
plt.xlabel('Year')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig(os.path.join(current_dir, 'tech_improvement.png'))
print("Global technology improvement plot saved in /EDA.")