import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('checkpoints\\CNN_BiLSTM_CTC\\losses.csv')

plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Loss'], marker='o', color='b', linestyle='-', markersize=5)
plt.title('Loss vs Epoch (CNN-BiLSTM-CTC)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True)

output_dir = 'results\\CNN_BiLSTM_CTC'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'loss_vs_epoch.png'))

plt.show()