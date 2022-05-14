with open('train_dataset.csv', 'r') as inf, open('train_dataset.csv', 'w') as out:
    for line in inf:
        if line.strip():
            out.write(line)
