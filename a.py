import pickle

# Load the preprocessing pipeline
preprocessing_pipe = pickle.load(open('pipepreprocessing.pkl', 'rb'))

# Check the transformers in the pipeline
for step_name, transformer, columns in preprocessing_pipe.steps:
    print(f"Step: {step_name}, Transformer: {type(transformer).__name__}, Columns: {columns}")
