"""Define ctc_decode function."""
# from tensorflow.python.ops import ctc_ops  # pylint: disable=no-name-in-module
import torch


def ctc_decode(y_pred, input_lengths, max_output_length, blank_idx=None):
    """
    Cut down from https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py#L4170

    Decodes the output of a softmax.
    Uses greedy (best path) search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the LogSoftMax.
        input_lengths: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        max_output_length: int giving the max output sequence length

    # Returns
        List: list of one element that contains the decoded sequence.
    """

    batch_size, time_steps, n_class = y_pred.shape
    if blank_idx is None:
        blank_idx = n_class - 1
    # y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())
    # input_lengths = tf.to_int32((tf.squeeze(input_lengths, axis=-1)))
    input_lengths = input_lengths.squeeze(-1).to(torch.int32)

    # (decoded, _) = ctc_ops.ctc_greedy_decoder(inputs=y_pred, sequence_length=input_lengths)
    
    max_length = max_output_length + 2  # giving 2 extra characters for CTC leeway

    # Step: Max
    y_pred = y_pred.argmax(dim=2) # -> (batch, time_steps)
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

    # sparse = decoded[0]
    # decoded_dense = tf.sparse_to_dense(sparse.indices, sparse.dense_shape, sparse.values, default_value=-1)

    # # Unfortunately, decoded_dense will be of different number of columns, depending on the decodings.
    # # We need to get it all in one standard shape, so let's pad if necessary.
    # cols = tf.shape(decoded_dense)[-1]

    # def pad():
    #     return tf.pad(decoded_dense, [[0, 0], [0, max_length - cols]], constant_values=-1)

    # def noop():
    #     return decoded_dense

    # return tf.cond(tf.less(cols, max_length), pad, noop)
