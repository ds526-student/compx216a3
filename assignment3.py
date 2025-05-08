import re
import math
import random
import collections

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.

    # counts the number of times each word appears in the sequence
    word_count = collections.Counter(sequence)

    # calculates the total number of each word in the sequence
    model = {word: count for word, count in word_count.items()}

    # return model
    return model

def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.

    # counts the number of times a bigram appears in the sequence
    bigram_count = collections.Counter(zip(sequence, sequence[1:]))

    # calculates the total number of each bigram in the sequence
    model = {bigram: count for bigram, count in bigram_count.items()}

    return model

def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    
    # counts the number of times an n-gram appears in the sequence
    n_gram_count = collections.Counter(zip(*([sequence[i:] for i in range(n)])))

    # calculates the total number of each n-gram in the sequence
    model = {n_gram: count for n_gram, count in n_gram_count.items()}

    return model

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    raise NotImplementedError

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor
    comp = 1 - factor
    for pred in preds[:-1]:
        if pred:
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            mult = comp * factor
            comp -= mult
    pred = preds[-1]
    mult += comp
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    
    # Task 1.1 test code
    model = build_unigram(sequence[:20])
    print("Unigram Model:")
    print(model)
    
    
    # Task 1.2 test code
    model = build_bigram(sequence[:20])
    print("\nBigram Model:")
    print(model)

    
    # Task 1.3 test code
    n_gram = 5
    model = build_n_gram(sequence[:20], n_gram)
    print("\n" + str(n_gram) + "-gram Model:")
    print(model)
    

    # Task 2 test code
    '''
    print(query_n_gram(model, tuple(sequence[:4])))
    '''

    # Task 3 test code
    '''
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    '''

    # Task 4.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 4.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''
