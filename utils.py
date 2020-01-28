import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from datetime import datetime

def mask_data(df, n, _seed ,ignore=[]):
    df = df.copy()
    vals = df.values

    # Create a list of features names without the ignored-features
    f_names = list(df.columns)
    f_without_ignore = [x for x in f_names if x not in ignore]
    all_data_filling = df.values
    masked_data_filling = {}
    masked_vals = []

    # mask each row
    for i, row in enumerate(vals):
        # Pick n features without replacement
        np.random.seed(_seed)
        feats = list(np.random.choice(f_without_ignore, size=n, replace=False))
        masked_data_filling[i] = {}
        for f in feats:
            masked_data_filling[i][f] = all_data_filling[i][f]
        row[feats] = None
        masked_vals.append(row)

    # Construct a DataFrame from the masked values and return
    masked_data = pd.DataFrame(masked_vals, columns=f_names)

    return masked_data, masked_data_filling


def _current_time_string():
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y %H-%M-%S")
    return date_time


def save_results(acc_lst, method, parameters, _seed, title):
    title += ' - accuracy per feature acquisition'
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.plot(range(len(acc_lst)), acc_lst, label=title)

    plt.xlabel('Used Features')
    plt.ylabel('Accuracy')
    # plt.xticks(range(0, len(aucs[0])), range(0, len(aucs[0])))

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(1))


    plt.title(title)
    now_str = _current_time_string()
    fig.savefig('results//' + title + ' ' + now_str + '.png')
    plt.close(fig)
    f = open('results//' + title + ' ' + now_str + '.txt', 'a')
    f.write('Method: ' + method + '\n')
    f.write('Parameters: ' + str(parameters) + '\n')
    f.write('Seed: ' + str(_seed) + '\n')
    f.write('Accuracy after each feature acquisition: ' + str(acc_lst) + '\n')
    f.close()

