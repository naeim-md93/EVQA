import os
import matplotlib.pyplot as plt

from src.utils.pyutils import load_file

history = load_file(path=os.path.join(os.getcwd(), 'History_E20.pickle'))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axes[0].plot(history['train_accuracy'], label='Train Accuracy')
axes[0].plot(history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Accuracy')
axes[0].grid()
axes[0].legend()

axes[1].plot(history['train_loss'], label='Train Loss')
axes[1].plot(history['val_loss'], label='Val Loss')
axes[1].set_title('Loss')
axes[1].grid()
axes[1].legend()

plt.show()
