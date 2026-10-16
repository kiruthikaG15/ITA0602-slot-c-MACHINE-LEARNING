import csv

data = [
    ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast", "EnjoySport"],
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

with open("enjoysport.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file created successfully!")

code:
import csv

# Read CSV file
with open("enjoysport.csv", "r") as file:
    data = list(csv.reader(file))

# Remove header
header = data[0]
data = data[1:]

num_attributes = len(data[0]) - 1

# Initial Specific and General hypotheses
S = ['0'] * num_attributes
G = [['?'] * num_attributes]

print("Initial S:", S)
print("Initial G:", G)

# Candidate-Elimination
for row in data:

    attributes = row[:-1]
    target = row[-1]

    if target == "Yes":

        # Generalize S
        for i in range(num_attributes):

            if S[i] == '0':
                S[i] = attributes[i]

            elif S[i] != attributes[i]:
                S[i] = '?'

        # Remove inconsistent hypotheses from G
        G = [
            g for g in G
            if all(g[i] == '?' or g[i] == attributes[i]
                   for i in range(num_attributes))
        ]

    else:

        # Specialize G
        new_G = []

        for g in G:

            if all(g[i] == '?' or g[i] == attributes[i]
                   for i in range(num_attributes)):

                for i in range(num_attributes):

                    if S[i] != '?' and S[i] != attributes[i]:
                        new_g = g.copy()
                        new_g[i] = S[i]

                        if new_g not in new_G:
                            new_G.append(new_g)

            else:
                new_G.append(g)

        G = new_G

print("\nFinal Specific Boundary (S):")
print(S)

print("\nFinal General Boundary (G):")
for g in G:
    print(g)
