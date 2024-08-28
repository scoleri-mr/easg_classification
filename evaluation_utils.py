import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

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
    
def compare_statistics(list1, list2, names_list, list1_name:str='train', list2_name:str='validation',  stat:str = 'verb', other=False):
    ''' function used to compare train and validation statistics or train and samples from diffusion models''' 
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

    # Plot the bar charts separately
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hist1 = ax1.bar(names1, percentages1, color='cornflowerblue')
    ax1.set_title(f'Top 10 {stat} frequencies in {list1_name}')
    ax1.set_xlabel(f'{stat}')
    ax1.set_ylabel('Percentage')

    hist2 = ax2.bar(names2, percentages2, color='rosybrown')
    ax2.set_title(f'Top 10 {stat} frequencies in {list2_name}')
    ax1.set_xlabel(f'{stat}')
    ax2.set_ylabel('Percentage')

    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and show plot
    plt.tight_layout()

    plt.savefig(f'perc_{stat}_{other}.jpg', format='jpg', dpi=500)
    plt.show()

    # Compute histogram distances and save them
    euclidean_dist = euclidean_distance(hist1, hist2)
    manhattan_dist = manhattan_distance(hist1, hist2)
    cosine_dist = cosine_distance(hist1, hist2)
    correlation_dist = correlation_distance(hist1, hist2)
    distances = [euclidean_dist, manhattan_dist, cosine_dist, correlation_dist]

    print(f"Euclidean Distance between {list1_name} and {list2_name} for {stat}:", euclidean_dist)
    print(f"Manhattan Distance {list1_name} and {list2_name} for {stat}:", manhattan_dist)
    print(f"Cosine Distance {list1_name} and {list2_name} for {stat}:", cosine_dist)
    print(f"Correlation Distance {list1_name} and {list2_name} for {stat}:", correlation_dist)
    return hist1, hist2, distances

def euclidean_distance(hist1, hist2):
    # Get the heights of the bars
    heights1 = np.array([rect.get_height() for rect in hist1])
    heights2 = np.array([rect.get_height() for rect in hist2])
    
    # Compute the Euclidean distance
    if len(heights1) != len(heights2):
        m = min(len(heights1),len(heights2))
        print("Warning: histograms have different lenghts. Truncating the longer one. Potential loss of information.")
        distance = np.linalg.norm(heights1[:m] - heights2[:m])
    else:
        distance = np.linalg.norm(heights1 - heights2)
    return distance

def manhattan_distance(hist1, hist2):
    # Get the heights of the bars
    heights1 = np.array([rect.get_height() for rect in hist1])
    heights2 = np.array([rect.get_height() for rect in hist2])
    
    # Compute the Manhattan distance
    if len(heights1) != len(heights2):
        m = min(len(heights1),len(heights2))
        print("Warning: histograms have different lenghts. Truncating the longer one. Potential loss of information.")
        distance = np.sum(np.abs(heights1[:m] - heights2[:m]))
    else:
        distance = np.sum(np.abs(heights1 - heights2))
    return distance

def cosine_distance(hist1, hist2):
    # Get the heights of the bars
    heights1 = np.array([rect.get_height() for rect in hist1])
    heights2 = np.array([rect.get_height() for rect in hist2])
    
    # Compute the cosine distance
    if len(heights1) != len(heights2):
        m = min(len(heights1),len(heights2))
        print("Warning: histograms have different lenghts. Truncating the longer one. Potential loss of information.")
        distance = cosine(heights1[:m], heights2[:m])
    else:
        distance = cosine(heights1, heights2)
    return distance

def correlation_distance(hist1, hist2):
    # Get the heights of the bars
    heights1 = np.array([rect.get_height() for rect in hist1])
    heights2 = np.array([rect.get_height() for rect in hist2])
    
    # Compute the Pearson correlation and then the distance
    if len(heights1) != len(heights2):
        m = min(len(heights1),len(heights2))
        print("Warning: histograms have different lenghts. Truncating the longer one. Potential loss of information.")
        correlation, _ = pearsonr(heights1[:m], heights2[:m])
    else:
        correlation, _ = pearsonr(heights1, heights2)
    distance = 1 - correlation
    return distance