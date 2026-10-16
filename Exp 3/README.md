
import pandas as pd
import numpy as np
from pprint import pprint

# Step 1: Create Dataset
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

columns = ['Outlook', 'Temperature', 'Humidity',
           'Wind', 'PlayTennis']

df = pd.DataFrame(data, columns=columns)

print("Training Dataset:")
print(df)

# Step 2: Calculate Entropy
def entropy(data):
    probabilities = data['PlayTennis'].value_counts(normalize=True)

    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Step 3: Calculate Information Gain
def information_gain(data, attribute):

    total_entropy = entropy(data)
    weighted_entropy = 0

    for value in data[attribute].unique():

        subset = data[data[attribute] == value]

        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# Step 4: ID3 Algorithm
def id3(data, attributes):

    target = data['PlayTennis']

    # If all outputs are same
    if len(target.unique()) == 1:
        return target.iloc[0]

    # If no attributes remain
    if len(attributes) == 0:
        return target.mode()[0]

    # Calculate information gain
    gains = {
        attr: information_gain(data, attr)
        for attr in attributes
    }

    # Select attribute with highest gain
    best = max(gains, key=gains.get)

    tree = {best: {}}

    for value in data[best].unique():

        subset = data[data[best] == value]

        remaining = [a for a in attributes if a != best]

        tree[best][value] = id3(subset, remaining)

    return tree

# Step 5: Build Decision Tree
attributes = list(df.columns[:-1])

tree = id3(df, attributes)

print("\nInformation Gain:")
for attr in attributes:
    print(attr, ":", round(information_gain(df, attr), 4))

print("\nDecision Tree:")
pprint(tree)

# Step 6: Predict New Sample
def predict(tree, sample):

    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))

    value = sample[attribute]

    if value not in tree[attribute]:
        return "Unknown"

    return predict(tree[attribute][value], sample)

# Step 7: New Sample
sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

result = predict(tree, sample)

print("\nNew Sample:", sample)
print("Predicted Result:", result)

