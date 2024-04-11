#Mostly not-original, original sections 
import numpy as np
from tqdm.notebook import tqdm

#This function takes the .txt file with all of the mario levels,
#splits the file into a separate string for each level,
#finds the number of unique characters in all the levels (this creates the vocab used for one-hot encoding),
#converts each character to its integer encoding,
#finds every seq_length sequence in each level,
#creates the inputs and targets from these sequence (where the target sequence would be the sequence following the input sequence),
#One-hot encodes the inputs,
#then outputs the char-to-int and int-to-char mappings, the vocab_size, the inputs, and the targets
def get_inputs_and_targets(corpus_txt_fpath, seq_length, snaking = False, pathing = True, column_depth = False):

    data = open(corpus_txt_fpath, 'r').read()

    #Original Section
    #If using column depth markers or snaking, modify the input to include them
    #I do this here instead of in 01_preprocess_data so that snaking and column_depth are among the hyperparameters
    if column_depth or not pathing or snaking:
        #Get every level's data
        level_array = data.split(")")[:-1]
        for i in range(len(level_array)):
            #Split the level string into an array of strings, one per column
            data_array = level_array[i].split("\n")[1:-1]
            #If using column_depth, add column characters every 5 columns
            if column_depth:
                for j in range(0, len(data_array), 5):
                    if j + 4 < len(data_array):  # Ensure index doesn't go out of bounds
                        data_array[j + 4] += ('d'*((j//5)+1))
            #If not using path information, remove the paths
            if not pathing:
                data_array = [column.replace('x', '-') for column in data_array]
            #If snaking the input, reverse every other column
            if snaking:
                for j in range(0, len(data_array), 2):
                    data_array[j] = data_array[j][::-1]
            #Join the columns back into a single string with newline characters (these mark the end of a column for the LSTM)
            level_array[i] = "\n".join(data_array)
        #Add all of the level strings back together
        data = "\n)\n".join(level_array)
        
    #print(data)
    # ==================================================
    
    # a list of strings, one for each level
    level_strs = data.strip().split(')')[:-1]  # last item is an empty string
    # ==================================================
    
    chars = []
    for level_str in level_strs:
        chars.extend(list(level_str))
    chars = sorted(list(set(chars)))
    vocab_size = len(chars)
    print('Unique chars:', chars)
    print('Number of unique chars:', vocab_size)
    
    # ==================================================
    
    # create two dictionaries
    char_to_ix = { ch:i for i, ch in enumerate(chars) }
    ix_to_char = { i:ch for i, ch in enumerate(chars) }
    
    # ==================================================
    
    # each level_array is an 1-d array of integers
    level_arrays = []
    for level_str in level_strs:
        level_arrays.append(np.array([char_to_ix[char] for char in list(level_str)]))
    # ==================================================
    def get_inputs_and_targets_from_level_array(level_array):
    
        inputs, targets = [], []

        for i in range(len(level_array) - seq_length):
            inputs.append(level_array[i:i+seq_length])
            targets.append(level_array[i+1:i+seq_length+1])

        inputs, targets = map(np.array, [inputs, targets])
        inputs = np.eye(vocab_size, dtype = 'float32')[inputs]

        return inputs, targets
    
    inputs, targets = [], []
    for level_array in tqdm(level_arrays, leave=False):
        inputs_temp, targets_temp = get_inputs_and_targets_from_level_array(level_array)
        inputs.extend(inputs_temp); targets.extend(targets_temp)
    inputs, targets = map(np.array, [inputs, targets])
    
    # # ==================================================
    
    return char_to_ix, ix_to_char, vocab_size, inputs, targets