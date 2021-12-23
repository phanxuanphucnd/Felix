import pickle
import timeit

with open('./ngram.pkl', 'rb') as f:
    ngram_lm = pickle.load(f)

text = input(f"Please enter a string: \n")

now = timeit.default_timer()

textout, prob, candidates = ngram_lm.generate(text=text, n_mask=3, k=5)

end = timeit.default_timer()

print(f"Text input  : {text}")
print(f"Text output : {textout}")
print(f"Prob        : {prob}")
print(f"Candidates  : {candidates}")
print(f"Time        : {(end - now)*1000} ms.")
