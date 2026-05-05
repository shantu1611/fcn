import matplotlib.pyplot as plt

epochs     = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480]
losses     = [0.5254, 0.1976, 0.1414, 0.1204, 0.1054, 0.0957, 0.0762, 0.0732, 0.0822, 0.0737, 0.0588, 0.0653, 0.0670, 0.0583, 0.0564, 0.0582, 0.0543, 0.0494, 0.0552, 0.0456, 0.0455, 0.0447, 0.0475, 0.0459]
accuracies = [0.9303, 0.9606, 0.9818, 0.9818, 0.9848, 0.9818, 0.9848, 0.9879, 0.9879, 0.9909, 0.9939, 0.9909, 0.9909, 0.9939, 0.9909, 0.9909, 0.9939, 0.9909, 0.9879, 0.9909, 0.9879, 0.9939, 0.9939, 0.9879]

FIG_W, FIG_H = 5.0, 3.5
FONT_SIZE = 9

plt.rcParams.update({
    'font.size'        : FONT_SIZE,
    'font.family'      : 'Times New Roman',
    'axes.titlesize'   : FONT_SIZE,
    'axes.labelsize'   : FONT_SIZE,
    'xtick.labelsize'  : FONT_SIZE - 1,
    'ytick.labelsize'  : FONT_SIZE - 1,
    'legend.fontsize'  : FONT_SIZE - 1,
    'lines.linewidth'  : 1.2,
    'lines.markersize' : 4,
})

# --- Epoch vs Accuracy ---
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(epochs, accuracies, color='#1f77b4', marker='o', label='Accuracy')
ax.set_title('Epoch vs Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.90, 1.01)
ax.grid(True, linewidth=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig('epoch_accuracy_ieee.pdf', bbox_inches='tight')
plt.show()

# --- Epoch vs Loss ---
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(epochs, losses, color='#d62728', marker='s', label='Loss')
ax.set_title('Epoch vs Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(True, linewidth=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig('epoch_loss_ieee.pdf', bbox_inches='tight')
plt.show()