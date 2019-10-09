"""Define ctc_decode function."""
import torch

def ctc_decode(y_pred, input_lengths, max_output_length, blank_idx=None):
    """

    Decodes the output of a softmax.
    Uses greedy (best path) search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the LogSoftMax.
        input_lengths: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        max_output_length: int giving the max output sequence length

    # Returns
        List: a batch list of one element that contains the decoded sequence.
    """

    batch_size, time_steps, n_class = y_pred.shape
    if blank_idx is None:
        blank_idx = n_class - 1
    
    max_length = max_output_length + 2  # giving 2 extra characters for CTC leeway

    # Step: Max
    y_pred = y_pred.argmax(dim=2) # (B, T)
    # Step: Merge and remove blank token, and cap on the max_output_length
    pred_idx = []
    for i, seq in enumerate(y_pred):
        if input_lengths[i] == 0:
            pred_idx.append([])
            continue
        len_seq = min(len(seq), input_lengths[i]) # only takes the first len_seq char
        seq = [seq[0].item()] + [seq[j].item() for j in range(1, len_seq) if seq[j] != seq[j-1]]
        seq = [char_idx for char_idx in seq if char_idx!=blank_idx][:max_length]
        pred_idx.append(seq)
    # NOTE no need to pad for output
    return pred_idx
