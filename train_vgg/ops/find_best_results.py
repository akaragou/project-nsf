import os
import numpy as np
from glob import glob


main_dir = '/media/data_cifs/clicktionary/clickme_experiment/results'
# main_dir = '/media/data/clickme/results/results'
val_pointer = 'validation_accuracies.npz'
folders = glob(main_dir + '/*')

t1_accs = []
t5_accs = []
names = []
for fo in folders:
    it_path = os.path.join(fo, val_pointer)
    # if os.path.exists(it_path):
    try:
        hopper = np.load(it_path)
        t1_accs.append(hopper['top_1'])
        t5_accs.append(hopper['top_5'])
        names.append(fo)
    except:
        print 'Skipping %s' % fo

best_t1_file = np.argsort(np.asarray([np.max(x) for x in t1_accs]))[::-1]
best_t5_file = np.argsort(np.asarray([np.max(x) for x in t5_accs]))[::-1]

best_t1_ckpt = [names[best_t1_file[0]], np.argmax(t1_accs[best_t1_file[0]])]
best_t5_ckpt = [names[best_t5_file[0]], np.argmax(t5_accs[best_t5_file[0]])]

print 'Best top-1 %s' % np.max(np.asarray([np.max(x) for x in t1_accs]))
print 'Best top-5 %s' % np.max(np.asarray([np.max(x) for x in t5_accs]))

np.savez('best_checkpoints', t1=best_t1_ckpt, t5=best_t5_ckpt)
