import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from collections import Counter
import numpy as np

def find_checkpoint(parent_folder, end):
    '''Give this function the last 4 digits of the experiment name and the folder in which to look for it.
    It will return the path to the checkpoint .ckpt'''
    import os
    subfolders = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]
    matching_subfolders = [subfolder for subfolder in subfolders if subfolder.endswith(str(end))]
    
    if len(matching_subfolders)>0:
        return parent_folder+matching_subfolders[0]+'/checkpoints/last.ckpt'
    else:
        return None

def count_predicted_triplets(triplets):
    '''Given the tiplets predictions from a model (or the original ones from the dataset), 
    counts the percentage of triplets that are predicted with zero, one, two, and three elements'''
    # Initialize counters
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0

    # Loop through triplets_diffusion once
    for el in triplets:
        triplet_size = el.size(0)
        if triplet_size == 0:
            count_0 += 1
        elif triplet_size == 1:
            count_1 += 1
        elif triplet_size == 2:
            count_2 += 1
        elif triplet_size == 3:
            count_3 += 1

    # Total number of elements
    total = len(triplets)

    # Print the percentages
    empty = count_0 * 100 / total
    one = count_1 * 100 / total
    two = count_2 * 100 / total
    three = count_3 * 100 / total
    print(f'{empty:.2f} % of graphs result in empty triplets')
    print(f'{one:.2f} % of graphs have 1 triplet')
    print(f'{two:.2f} % of graphs have 2 triplets')
    print(f'{three:.2f} % of graphs have 3 triplets')

    return [empty, one, two, three]

def get_num_nodes(triplets1, triplets2, pred_name='Diffusion'):
    '''Function to plot the comparison between the number of nodes in the original dataset
    and the number of nodes in model-generated samples'''
    import pandas as pd
    import matplotlib.pyplot as plt
    # Get the number of objects in each training and validation sample
    num_objects1 = [triplets1[i].size(0) for i in range(len(triplets1))]
    num_objects2 = [triplets2[i].size(0) for i in range(len(triplets2))]

    # Convert to pandas Series
    objects1_pd = pd.Series(num_objects1, name='Train')
    objects2_pd = pd.Series(num_objects2, name=pred_name)

    # Get value counts and sort by index
    objects1_count = objects1_pd.value_counts().sort_index()
    objects1_count = objects1_count*100/objects1_count.sum()

    objects2_count = objects2_pd.value_counts().sort_index()
    objects2_count = objects2_count*100/objects2_count.sum()

    # Combine both Series into a DataFrame
    objects_count_df = pd.DataFrame({'Original': objects1_count, pred_name: objects2_count}).fillna(0)

    objects_count_df.plot.bar()
    plt.xticks(rotation=0)
    plt.title(f'Number of Objects in original dataset and {pred_name} samples')
    plt.xlabel('Number of Objects')
    plt.ylabel('Frequency')
    plt.show()
    
def compare_statistics(list1, list2, list3, names_list, list1_name:str='train', list2_name:str='validation', list3_name:str='vae', stat:str = 'verb', other=False, save=False):

    ''' function used to compare train and validation statistics or train and samples from diffusion models/VAE''' 
    import matplotlib.pyplot as plt
    from collections import Counter
    import pandas as pd

    # Create Counters for both lists
    c1 = Counter(list1)
    c2 = Counter(list2)

    # Extract the top 10 elements from each list
    top10_1 = c1.most_common(10)
    top10_2 = c2.most_common(10)

    if other:
        # Calculate the sum of all other
        other_count_1 = sum(c1.values()) - sum(count for _,count in top10_1)
        other_count_2 = sum(c2.values()) - sum(count for _,count in top10_2)

        top10_1.append(('Other', other_count_1))
        top10_2.append(('Other', other_count_2))

    # Get the total counts for calculating percentages
    total_count_1 = sum(c1.values())
    total_count_2 = sum(c2.values())

    # Get the elements names and their percentages
    names1 = [names_list[el[0]] if el[0] != 'Other' else 'Other' for el in top10_1]
    percentages1 = [el[1] / total_count_1 * 100 for el in top10_1]
    names2 = [names_list[el[0]] if el[0] != 'Other' else 'Other' for el in top10_2]
    percentages2 = [el[1] / total_count_2 * 100 for el in top10_2]

    c3 = Counter(list3)
    top10_3 = c3.most_common(10)

    if other:
        other_count_3 = sum(c3.values()) - sum(count for _,count in top10_3)
        top10_3.append(('Other', other_count_3))

    total_count_3 = sum(c3.values())

    names3 = [names_list[el[0]] if el[0] != 'Other' else 'Other' for el in top10_3]
    percentages3 = [el[1] / total_count_3 * 100 for el in top10_3]

    # Plot the bar charts separately
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))


    hist1 = ax1.bar(names1, percentages1, color='cornflowerblue')
    ax1.set_title(f'Top 10 {stat} frequencies in {list1_name}')
    ax1.set_xlabel(f'{stat}')
    ax1.set_ylabel('Percentage')

    hist2 = ax2.bar(names2, percentages2, color='rosybrown')
    ax2.set_title(f'Top 10 {stat} frequencies in {list2_name}')
    ax1.set_xlabel(f'{stat}')
    ax2.set_ylabel('Percentage')

    hist3 = ax3.bar(names3, percentages3, color='lightgreen')
    ax3.set_title(f'Top 10 {stat} frequencies in {list3_name}')
    ax3.set_xlabel(f'{stat}')
    ax3.set_ylabel('Percentage')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and show plot
    plt.tight_layout()

    if save: plt.savefig(f'perc_{stat}_{other}.jpg', format='jpg', dpi=500)
    plt.show()
    return c1, c2

def compare_statistics_v2(list1, list2, list3, names_list, list1_name:str='train', list2_name:str='validation', list3_name:str='vae', stat:str = 'verb', other=False, save=False):

    ''' function used to compare train and validation statistics or train and samples from diffusion models/VAE''' 
    import matplotlib.pyplot as plt
    from collections import Counter
    import pandas as pd

    # Create Counters for both lists
    c1 = Counter(list1)
    c2 = Counter(list2)

    # Extract the top 10 elements from each list
    top10_1 = c1.most_common(10)
    top10_2 = c2.most_common(10)

    if other:
        # Calculate the sum of all other
        other_count_1 = sum(c1.values()) - sum(count for _,count in top10_1)
        other_count_2 = sum(c2.values()) - sum(count for _,count in top10_2)

        top10_1.append(('Other', other_count_1))
        top10_2.append(('Other', other_count_2))

    # Get the total counts for calculating percentages
    total_count_1 = sum(c1.values())
    total_count_2 = sum(c2.values())

    # Get the elements names and their percentages
    names1 = [names_list[el[0]] if el[0] != 'Other' else 'Other' for el in top10_1]
    percentages1 = [el[1] / total_count_1 * 100 for el in top10_1]
    names2 = [names_list[el[0]] if el[0] != 'Other' else 'Other' for el in top10_2]
    percentages2 = [el[1] / total_count_2 * 100 for el in top10_2]

    c3 = Counter(list3)
    top10_3 = c3.most_common(10)

    if other:
        other_count_3 = sum(c3.values()) - sum(count for _,count in top10_3)
        top10_3.append(('Other', other_count_3))

    total_count_3 = sum(c3.values())

    names3 = [names_list[el[0]] if el[0] != 'Other' else 'Other' for el in top10_3]
    percentages3 = [el[1] / total_count_3 * 100 for el in top10_3]

    max_height = max(max(percentages1), max(percentages2), max(percentages3))


    # Plot the bar charts separately
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))


    hist1 = ax1.bar(names1, percentages1, color='cornflowerblue')
    ax1.set_title(f'Top 10 {stat} frequencies in {list1_name}')
    # ax1.set_xlabel(f'{stat}')
    ax1.set_ylabel('Percentage')

    hist2 = ax2.bar(names2, percentages2, color='rosybrown')
    ax2.set_title(f'Top 10 {stat} frequencies in {list2_name}')
    # ax1.set_xlabel(f'{stat}')
    # ax2.set_ylabel('Percentage')

    hist3 = ax3.bar(names3, percentages3, color='teal')
    ax3.set_title(f'Top 10 {stat} frequencies in {list3_name}')
    # ax3.set_xlabel(f'{stat}')
    # ax3.set_ylabel('Percentage')
    
    ax1.set_ylim(0, max_height+3)
    ax2.set_ylim(0, max_height+3)
    ax3.set_ylim(0, max_height+3)


    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and show plot
    plt.tight_layout()

    if save: plt.savefig(f'perc_{stat}_{other}.jpg', format='jpg', dpi=500)
    plt.show()
    return

def top10_distances(c1, c2):
    '''Function to compute distances for top 10 elements in each counter'''
    # Extract top 10 elements for each counter
    top10_1 = dict(c1.most_common(10))
    top10_2 = dict(c2.most_common(10))
    
    # Ensure the keys match between the two dictionaries
    keys = set(top10_1.keys()).union(set(top10_2.keys()))
    for key in keys:
        top10_1.setdefault(key, 0)
        top10_2.setdefault(key, 0)
    
    # turn values into percentages
    s1 = sum(list(c1.values()))
    s2 = sum(list(c2.values()))

    for k,v in top10_1.items():
        top10_1[k] = v/s1

    for k,v in top10_2.items():
        top10_2[k] = v/s2

    # Calculate Euclidean and Manhattan distances for top 10
    ed = euclidean_distance(Counter(top10_1), Counter(top10_2))
    # md = manhattan_distance(Counter(top10_1), Counter(top10_2))
    print(f"top10 euclidean distance: {ed}")
    # print(f"top10 manhattan distance: {ed}")
    return ed

def all_distances(c1, c2):
    d1 = dict(c1)
    d2 = dict(c2)
    keys = set(d1.keys()).union(set(d2.keys()))
    for key in keys:
        d1.setdefault(key, 0)
        d2.setdefault(key, 0)

    # turn values into percentages
    s1 = sum(list(d1.values()))
    s2 = sum(list(d2.values()))

    for k,v in d1.items():
        d1[k] = v/s1

    for k,v in d2.items():
        d2[k] = v/s2

    ed_all = euclidean_distance(Counter(d1), Counter(d2))
    # md_all = manhattan_distance(Counter(d1), Counter(d2))
    print(f"complete euclidean distance: {ed_all}")
    # print(f"complete manhattan distance: {ed_all}")
    return ed_all

def euclidean_distance(c1, c2):
    '''Function to compute Euclidean distance between two counters'''
    # Extract keys and values for both counters
    keys = list(c1.keys())
    v1 = np.array([c1[k] for k in keys])
    v2 = np.array([c2[k] for k in keys])
    # Calculate Euclidean distance
    euclidean_distance = np.sqrt(np.sum((v1 - v2) ** 2))
    return euclidean_distance

def manhattan_distance(c1, c2):
    '''Function to compute Manhattan distance between two counters'''
    # Extract keys and values for both counters
    keys = list(c1.keys())
    v1 = np.array([c1[k] for k in keys])
    v2 = np.array([c2[k] for k in keys])
    # Calculate Manhattan distance
    manhattan_distance = np.sum(np.abs(v1 - v2))
    return manhattan_distance

def compare_statistics2(list1, list2, names_list, list1_name:str='train', list2_name:str='validation',  stat:str = 'verb', other=False, save=False):
    ''' function used to compare train and validation statistics or train and samples from diffusion models/VAE''' 
    import matplotlib.pyplot as plt
    from collections import Counter
    import pandas as pd
    import numpy as np

    # Create Counters for both lists
    c1 = Counter(list1)
    c2 = Counter(list2)

    # Extract the top 10 elements from each list
    top10_1 = c1.most_common(10)
    top10_2 = c2.most_common(10)

    if other:
        # Calculate the sum of all other
        other_count_1 = sum(c1.values()) - sum(count for _,count in top10_1)
        other_count_2 = sum(c2.values()) - sum(count for _,count in top10_2)

        top10_1.append(('Other', other_count_1))
        top10_2.append(('Other', other_count_2))

    # Create a set of unique elements from both top 10 lists
    unique_elements = set([el[0] for el in top10_1] + [el[0] for el in top10_2])

    # Get the total counts for calculating percentages
    total_count_1 = sum(c1.values())
    total_count_2 = sum(c2.values())

    # Create a dictionary to hold the percentage data for each element
    data1 = {el: c1[el] / total_count_1 * 100 for el in unique_elements}
    data2 = {el: c2[el] / total_count_2 * 100 for el in unique_elements}

    # Prepare the data for plotting
    names = [names_list[el] if el != 'Other' else 'Other' for el in unique_elements]
    percentages1 = [data1[el] for el in unique_elements]
    percentages2 = [data2[el] for el in unique_elements]

    # Sort the names and percentages by the first dataset (optional)
    sorted_indices = np.argsort(percentages1)[::-1]
    names = np.array(names)[sorted_indices]
    percentages1 = np.array(percentages1)[sorted_indices]
    percentages2 = np.array(percentages2)[sorted_indices]

    # Plot the data in a single bar chart with two bars per category
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35  # the width of the bars

    # Create the positions for the bars
    indices = np.arange(len(names))

    # Plotting both sets of data side by side
    ax.bar(indices - width/2, percentages1, width, label=list1_name, color='cornflowerblue')
    ax.bar(indices + width/2, percentages2, width, label=list2_name, color='rosybrown')

    # Add some labels and titles
    ax.set_xlabel(f'{stat}')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Comparison of Top 10 {stat} Frequencies')
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=45, ha='right')

    # Add a legend
    ax.legend()

    # Adjust layout and show plot
    plt.tight_layout()

    if save: plt.savefig(f'perc_{stat}_{other}.jpg', format='jpg', dpi=500)
    plt.show()

    return c1, c2
