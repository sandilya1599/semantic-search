def generate_training_set(id_seq, window_size):
    """
    Generate context and target IDs for CBOW model.
    """
    context_ids = []
    target_id = []
    for sequence in id_seq:
        for i in range(window_size, len(sequence) - window_size):
            context = sequence[i-window_size:i]
            context.extend(sequence[i+1:i+window_size+1])
            context_ids.append(context)
            target_id.append(sequence[i])
    print("Dataset Entries: ", len(context_ids))
    return context_ids, target_id

def generate_skip_gram_train_set(id_seq, window_size):
    """
    Generate context and target IDs for SkipGram model.
    """
    context_ids = []
    target_id = []
    for sequence in id_seq:
        for i in range(window_size, len(sequence) - window_size):
            context = sequence[i-window_size:i]
            context.extend(sequence[i+1:i+window_size+1])
            context_ids.append(context)
            target_id.append(sequence[i])
    return context_ids, target_id
