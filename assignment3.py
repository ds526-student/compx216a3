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

    # calculates the total number of each word in the sequence, and returns the model as a dictionary
    model = {word: {'count': {'value': count}} for word, count in word_count.items()}

    # return model
    return model

def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.
    # counts the number of times a bigram appears in the sequence
    bigram_count = collections.Counter(zip(sequence, sequence[1:]))

    # calculates the total number of each bigram in the sequence, and returns the model with an inner and outer dictionary
    model = {}
    for (word_1, word_2), count in bigram_count.items():
        if word_1 not in model:
            model[word_1] = {}
        model[word_1][word_2] = count

    return model

def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    
    # counts the number of times an n-gram appears in the sequence
    n_gram_count = collections.Counter(zip(*([sequence[i:] for i in range(n)])))

    # calculates the total number of each n-gram in the sequence, and returns the model with an inner and outer dictionary
    model = {}
    for n_gram, count in n_gram_count.items():
        prefix, next_word = n_gram[:-1], n_gram[-1]
        if prefix not in model:
            model[prefix] = {}
        model[prefix][next_word] = count

    return model

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.

    # check if sequence exists in the model
    if sequence in model:
        return model[sequence]
    else:
        return None

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

    # store the predictions
    predictions = [] 

    # iterate through the models
    for model in models:
        # find the n-gram size
        n_gram = max(len(key) for key in model.keys()) + 1

        # compare the length of the sequence to the n-gram size
        if len(sequence) >= n_gram - 1:
            context = tuple(sequence[-(n_gram - 1):])  # prediction context
            prediction = query_n_gram(model, context)  
            # if there is a prediction append it
            if prediction is not None:
                predictions.append(prediction)

    # return None if no predictions were made
    if not predictions:
        return None

    # blend the predictions
    blended_predictions = blended_probabilities(predictions)
    blended_keys = list(blended_predictions.keys())
    blended_values = list(blended_predictions.values())

    # token randomly chosen from blended probabilities
    return random.choices(blended_keys, weights=blended_values, k=1)[0]

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    
    # store the log likelihood
    log_likelihood = 0.0

    # iterate through the sequence
    for i, word in enumerate(sequence):
        # determine the model and context based on the index
        if i == 0:
            # if the sequence is empty, use the unigram
            model = models[-1] 
            context = ()
        elif i < len(models):
            # if the index is less than the number of models, use the corresponding model
            model = models[-(i + 1)]
            context = tuple(sequence[:i])
        else:
            # if the index is greater than the number of models, use the largest n-gram model
            model = models[0]  # largest n-gram
            context = tuple(sequence[i - (len(models) - 1):i])

        # query the model for the probabilities
        probs = query_n_gram(model, context)

        if probs is None or word not in probs:
            return -math.inf

        total = sum(probs.values()) # total probability
        prob = probs[word] / total # probability of the word
        log_likelihood += math.log(prob) # log of the probability

    return log_likelihood

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.

    # store the log likelihood
    log_likelihood = 0.0

    # iterate through the sequence
    for i, word in enumerate(sequence):
        # store the predictions
        preds = []
        # iterate through the models
        for model in models:
            # find the n-gram size
            n_gram = max(len(key) for key in model.keys()) + 1

            # compare the length of the sequence to the n-gram size
            if i >= n_gram - 1:
                context = tuple(sequence[i - (n_gram - 1):i])
            # if the index is less than the n-gram size, use the corresponding model
            else:
                context = tuple(sequence[:i])

            pred = query_n_gram(model, context) 

            # append the prediction to the list
            if pred is not None: 
                preds.append(pred)
        # check that there are predictions
        if not preds:
            return -math.inf
        
        blended = blended_probabilities(preds) # blend the predictions

        # check that the word is in the blended predictions
        if word not in blended or blended[word] == 0:
            return -math.inf
        
        log_likelihood += math.log(blended[word]) # log of the blended probability

    return log_likelihood

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
    print("\nQuerying n-gram model:")
    print(query_n_gram(model, tuple(sequence[:4])))

    # Task 3 test code
    print("\nSampling from blended predictions:")
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()

    # Task 4.1 test code
    print("\nLog likelihood of the sequence:")
    print(log_likelihood_ramp_up(sequence[:20], models))

    # Task 4.2 test code
    print("\nLog likelihood of the sequence with blended predictions:")
    print(log_likelihood_blended(sequence[:20], models))
