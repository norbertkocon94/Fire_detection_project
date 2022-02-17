# Results Plot: Accuracy and Loss metrics
import matplotlib.pyplot as plt
import pandas as pd

from src.model.model import model

results = pd.DataFrame(model.history.history)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(results['accuracy'], label='Training Accuracy', color='green', marker='*')
plt.plot(results['val_accuracy'], label='Training Accuracy', color='red', marker='*')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results['loss'], label='Training Loss', color='green', marker='*')
plt.plot(results['val_loss'], label='Val Loss', color='red', marker='*')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

