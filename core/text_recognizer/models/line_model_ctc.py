# """Define LineModelCtc class and associated functions."""
# from typing import Callable, Dict, Tuple

# import editdistance
# import numpy as np

# import torch
# from torch import nn

# from text_recognizer.datasets.dataset_sequence import DatasetSequence
# from text_recognizer.datasets import EmnistLinesDataset
# from text_recognizer.models.base import Model
# from text_recognizer.networks.line_lstm_ctc import line_lstm_ctc
# from text_recognizer.networks.ctc import ctc_decode

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class LineModelCtc(Model):
#     """Model for recognizing handwritten text in an image of a line, using CTC loss/decoding."""
#     def __init__(self,
#                  dataset_cls: type = EmnistLinesDataset,
#                  network_fn: Callable = line_lstm_ctc,
#                  dataset_args: Dict = None,
#                  network_args: Dict = None):
#         """Define the default dataset and network values for this model."""
#         default_dataset_args: dict = {}
#         if dataset_args is None:
#             dataset_args = {}
#         dataset_args = {**default_dataset_args, **dataset_args}

#         default_network_args = {'window_width': 12, 'window_stride': 5}
#         if network_args is None:
#             network_args = {}
#         network_args = {**default_network_args, **network_args}
#         super().__init__(dataset_cls, network_fn, dataset_args, network_args)
#         # self.batch_format_fn = format_batch_ctc
    

#     # TODO: need to change loss function (blank=0, and loss input args)
#     def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
#         if callbacks is None:
#             callbacks = []

#         train_sequence = DatasetSequence(
#             # dataset.x_train[:5], dataset.y_train[:5],  # @@@
#             dataset.x_train, dataset.y_train, 
#             batch_size=batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn
#         )

#         print(f"Total #training: {len(train_sequence.dataset)}")
#         print(f"Total #params: {sum([param.nelement() for param in self.network.parameters()])}")

#         self.network.to(device)
#         self.network.train()

#         optimizer_class = self.optimizer()
#         optimizer = optimizer_class(self.network.parameters(), lr=3e-4) # magic Adam lr
#         blank_idx = self.data.num_classes-1
#         loss_fn_class = self.loss()
#         loss_fn = loss_fn_class(blank=blank_idx)
        
#         validation_interval = 4
#         # total_loss = [] #@@@@
#         # validation_interval = 1 #@@@
#         for epoch in range(epochs):  # loop over the dataset multiple times
#             running_loss = 0.0
#             for i, batch in enumerate(train_sequence, 0):

#                 inputs, labels = batch
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 labels = torch.squeeze(labels) # shape (B, T)
#                 # NOTE this assumes blank_idx only occurs in continuity in the last part of seq
#                 output_lengths = (torch.sum(labels != blank_idx, dim=1)).to(torch.int32).to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 log_soft_max, input_lengths = self.network(inputs)
#                 loss = loss_fn(log_soft_max.cpu(), labels.cpu(), input_lengths.cpu(), output_lengths.cpu())
#                 loss.backward()
#                 optimizer.step()

#                 # print statistics
#                 running_loss += loss.item()
#                 # total_loss.append(loss.item()) #@@@@

#             # if epoch % interval == (interval-1):    # print every interval-epochs
#             print(f"[{epoch+1}, {i+1}] loss: {running_loss/(i+1):.5f}")

#             if epoch % validation_interval == (validation_interval-1):
#                 # score = self.evaluate(dataset.x_train[:5], dataset.y_train[:5], verbose=False) #@@@
#                 score = self.evaluate(dataset.x_test, dataset.y_test)
#                 print(f"Validation score: {score:.4f}")
        
#         print('Finished Training')
#         # import matplotlib.pyplot as plt #@@@@
#         # plt.plot(total_loss) #@@@@
#         # plt.show()

#     def loss(self):
#         # """Dummy loss function: just pass through the loss that we computed in the network."""
#         # return {'ctc_loss': lambda y_true, y_pred: y_pred}
#         return nn.CTCLoss

#     def metrics(self):
#         """We could probably pass in a custom character accuracy metric for 'ctc_decoded' output here."""
#         return None

#     def evaluate(self, x, y, batch_size: int = 16, verbose: bool = True) -> float:
#         output_length = self.data.output_shape[0]
#         test_sequence = DatasetSequence(x, y, batch_size, format_fn=self.batch_format_fn)
#         # network(x) -> y_pred
#         with torch.no_grad():
#             was_training = self.network.training
#             self.network.eval()
#             preds_raw = []
#             input_lengths = []
#             labels_raw = []
#             for i, batch in enumerate(test_sequence):
#                 batch_x, batch_y = map(lambda out: out.to(device), batch)
#                 batch_x = batch_x.to(device)
#                 batch_y = batch_y.to(device)
#                 # log_soft_max (T, B, num_classes)
#                 log_soft_max, batch_input_lengths = map(lambda out: out.to("cpu"), self.network(batch_x))
#                 preds_raw.append(log_soft_max.permute(1,0,2))
#                 input_lengths.append(batch_input_lengths)
#                 labels_raw.append(batch_y.to("cpu"))
#             # preds_raw: (B, T, C)
#             preds_raw, input_lengths = torch.cat(preds_raw), torch.cat(input_lengths)
#             labels_raw = torch.cat(labels_raw).numpy() # (batch, output_length)

        
#         # TODO make sure preds_raw: (batch, time_step, n_class)
#         # ctc_decode(y_pred) -> pred_idx: List[List]
#         # input requires (N, T, C)
#         preds = ctc_decode(preds_raw, input_lengths, output_length)

#         # We can use the `ctc_decoded` layer that is part of our model here.
#         # decoding_model = KerasModel(inputs=self.network.input, outputs=self.network.get_layer('ctc_decoded').output)
#         # preds = decoding_model.predict_generator(test_sequence)

#         # trues = np.argmax(y, -1)
#         trues = labels_raw
#         pred_strings = [''.join(self.data.mapping.get(label, '') for label in pred).strip(' |_') for pred in preds]
#         true_strings = [''.join(self.data.mapping.get(label, '') for label in true).strip(' |_') for true in trues]

#         char_accuracies = [
#             1 - editdistance.eval(true_string, pred_string) / len(true_string)
#             for pred_string, true_string in zip(pred_strings, true_strings)
#         ]
#         if verbose:
#             sorted_ind = np.argsort(char_accuracies)
#             print("\nLeast accurate predictions:")
#             for ind in sorted_ind[:5]:
#                 print(f'True: {true_strings[ind]}')
#                 print(f'Pred: {pred_strings[ind]}')
#             print("\nMost accurate predictions:")
#             for ind in sorted_ind[-5:]:
#                 print(f'True: {true_strings[ind]}')
#                 print(f'Pred: {pred_strings[ind]}')
#             print("\nRandom predictions:")
#             random_ind = np.random.randint(0, len(char_accuracies), 5)
#             for ind in random_ind:  # pylint: disable=not-an-iterable
#                 print(f'True: {true_strings[ind]}')
#                 print(f'Pred: {pred_strings[ind]}')
#         mean_accuracy = np.mean(char_accuracies)
        
#         if was_training:
#             self.network.train()

#         return mean_accuracy

#     def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
#         output_length = self.data.output_shape[0]
#         # softmax_output_fn = K.function(
#         #     [self.network.get_layer('image').input, K.learning_phase()],
#         #     [self.network.get_layer('softmax_output').output]
#         # )
#         if image.dtype == np.uint8:
#             image = (image / 255).astype(np.float32)

#         # Get the prediction and confidence using softmax_output_fn, passing the right input into it.
#         # Your code below (Lab 3)
#         input_image = np.expand_dims(image, 0)
#         with torch.no_grad():
#             was_training = self.network.training
#             self.network.eval()

#             input_image = torch.from_numpy(input_image).to(device)
#             y_pred, input_lengths = self.network(input_image) # y_pred (T,N,C)

#         # input_length = np.array([softmax_output.shape[1]])
#         # decoded, log_prob = K.ctc_decode(softmax_output, input_length, greedy=True)
#         pred_idx = ctc_decode(y_pred.permute((1,0,2)), input_lengths, output_length) # input requires (N,T,C)
#         pred_raw = pred_idx[0] # the batch only contains 1 element

#         # pred_raw = K.eval(decoded[0])[0]
#         pred = ''.join(self.data.mapping[label] for label in pred_raw).strip(' |_')

#         # neg_sum_logit = K.eval(log_prob)[0][0]
#         max_logit, _ = torch.max(y_pred.squeeze(dim=1), dim=1)
#         # TODO: is this the right confidence?
#         conf = np.exp(max_logit.sum())
#         # conf = np.exp(-neg_sum_logit)
        
#         # Your code above (Lab 3)
#         if was_training:
#             self.network.train()

#         return pred, conf


# # def format_batch_ctc(batch_x, batch_y):
# #     """
# #     Because CTC loss needs to be computed inside of the network, we include information about outputs in the inputs.
# #     """
# #     batch_size = batch_y.shape[0]
# #     y_true = np.argmax(batch_y, axis=-1)

# #     label_lengths = []
# #     for ind in range(batch_size):
# #         # Find all of the indices in the label that are blank
# #         empty_at = np.where(batch_y[ind, :, -1] == 1)[0]
# #         # Length of the label is the pos of the first blank, or the max length
# #         if empty_at.shape[0] > 0:
# #             label_lengths.append(empty_at[0])
# #         else:
# #             label_lengths.append(batch_y.shape[1])

# #     batch_inputs = {
# #         'image': batch_x,
# #         'y_true': y_true,
# #         'input_length': np.ones((batch_size, 1)),  # dummy, will be set to num_windows in network
# #         'label_length': np.array(label_lengths)
# #     }
# #     batch_outputs = {
# #         'ctc_loss': np.zeros(batch_size),  # dummy
# #         'ctc_decoded': y_true
# #     }
# #     return batch_inputs, batch_outputs

"""Define LineModelCtc class and associated functions."""
from typing import Callable, Dict, Tuple

import editdistance
import numpy as np

import torch
from torch import nn
from torch_baidu_ctc import CTCLoss

from text_recognizer.datasets.dataset_sequence import DatasetSequence
from text_recognizer.datasets import EmnistLinesDataset
from text_recognizer.models.base import Model
from text_recognizer.networks.line_lstm_ctc import line_lstm_ctc
from text_recognizer.networks.ctc import ctc_decode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu"s)
class LineModelCtc(Model):
    """Model for recognizing handwritten text in an image of a line, using CTC loss/decoding."""
    def __init__(self,
                 dataset_cls: type = EmnistLinesDataset,
                 network_fn: Callable = line_lstm_ctc,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        default_dataset_args: dict = {}
        if dataset_args is None:
            dataset_args = {}
        dataset_args = {**default_dataset_args, **dataset_args}

        default_network_args = {'window_width': 12, 'window_stride': 5}
        if network_args is None:
            network_args = {}
        network_args = {**default_network_args, **network_args}
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
        # self.batch_format_fn = format_batch_ctc
    

    # TODO: need to change loss function (blank=0, and loss input args)
    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        if callbacks is None:
            callbacks = []

        train_sequence = DatasetSequence(
            # dataset.x_train[:5], dataset.y_train[:5],  # @@@
            dataset.x_train, dataset.y_train, 
            batch_size=batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn
        )

        print(f"Total #training: {len(train_sequence.dataset)}")
        print(f"Total #params: {sum([param.nelement() for param in self.network.parameters()])}")

        self.network.to(device)
        self.network.train()

        optimizer_class = self.optimizer()
        optimizer = optimizer_class(self.network.parameters(), lr=3e-4) # magic Adam lr
        blank_idx = self.data.num_classes-1
#         loss_fn_class = self.loss()
#         loss_fn = loss_fn_class(blank=)
#         loss_fn = loss_fn_class(blank=blank_idx, reduction='sum', zero_infinity=True)
        loss_fn = CTCLoss(blank=blank_idx, reduction='mean', average_frames=True)

        
        validation_interval = 10
        # total_loss = [] #@@@@
#         validation_interval = 1 #@@@
        score = self.evaluate(dataset.x_test, dataset.y_test)
        print(f"Validation score: {score:.4f}")
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_count = 0
            for i, batch in enumerate(train_sequence, 0):

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
#                 labels = torch.squeeze(labels) # shape (B, T)
                # NOTE this assumes blank_idx only occurs in continuity in the last part of seq
#                 output_lengths = (torch.sum(labels != blank_idx, dim=1)).to(torch.long).to(device)
                # first get output lengths without padding, then calculate length, then concat the output
                output_lengths = (torch.sum(labels != blank_idx, dim=1)).to(torch.long).to(device)
                labels_concat = torch.cat([labels[i, :l] for i, l in enumerate(output_lengths)])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                log_soft_max, input_lengths = self.network(inputs)
                loss = loss_fn(log_soft_max.cpu(), labels_concat.cpu(), input_lengths.cpu(), output_lengths.cpu())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_count += len(inputs)
                # total_loss.append(loss.item()) #@@@@

            # if epoch % interval == (interval-1):    # print every interval-epochs
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/running_count:.5f}")

            if epoch % validation_interval == (validation_interval-1):
                # score = self.evaluate(dataset.x_train[:5], dataset.y_train[:5], verbose=False) #@@@
                score = self.evaluate(dataset.x_test, dataset.y_test)
                print(f"Validation score: {score:.4f}")
        
        print('Finished Training')
        # import matplotlib.pyplot as plt #@@@@
        # plt.plot(total_loss) #@@@@
        # plt.show()

    def loss(self):
        # """Dummy loss function: just pass through the loss that we computed in the network."""
        # return {'ctc_loss': lambda y_true, y_pred: y_pred}
        return nn.CTCLoss

    def evaluate(self, x, y, batch_size: int = 16, verbose: bool = True) -> float:
        blank_idx = self.data.num_classes - 1
        output_length = self.data.output_shape[0]
        test_sequence = DatasetSequence(x, y, batch_size, format_fn=self.batch_format_fn)
        # network(x) -> y_pred
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()
            preds_raw = []
            input_lengths = []
            labels_raw = []
            
            running_loss, running_count = 0, 0
            for i, batch in enumerate(test_sequence):
                batch_x, batch_y = map(lambda out: out.to(device), batch)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # log_soft_max (T, B, num_classes)
                log_soft_max, batch_input_lengths = map(lambda out: out.to("cpu"), self.network(batch_x))
                preds_raw.append(log_soft_max.permute(1,0,2))
                input_lengths.append(batch_input_lengths) # TODO
                labels_raw.append(batch_y.to("cpu"))
                output_lengths = (torch.sum(batch_y != blank_idx, dim=1)).to(torch.long).cpu() # TODO
                
                loss = CTCLoss(average_frames=True, blank=blank_idx, reduction='mean')(log_soft_max, batch_y.cpu(), batch_input_lengths, output_lengths)
                running_loss += loss.item()
                running_count += len(batch_x)
            # preds_raw: (B, T, C)
            preds_raw, input_lengths = torch.cat(preds_raw), torch.cat(input_lengths)
            labels_raw = torch.cat(labels_raw).numpy() # (batch, output_length)
        print(f"Validation loss: {running_loss/running_count:.4f}")
        print((torch.argmax(preds_raw, dim=2)!=79).sum())
        print(torch.argmax(preds_raw, dim=2)[0])
        
        # TODO make sure preds_raw: (batch, time_step, n_class)
        # ctc_decode(y_pred) -> pred_idx: List[List]
        # input requires (N, T, C)
        preds = ctc_decode(preds_raw, input_lengths, output_length)
#         print(preds)

        # We can use the `ctc_decoded` layer that is part of our model here.
        # decoding_model = KerasModel(inputs=self.network.input, outputs=self.network.get_layer('ctc_decoded').output)
        # preds = decoding_model.predict_generator(test_sequence)

        # trues = np.argmax(y, -1)
        trues = labels_raw
        pred_strings = [''.join(self.data.mapping.get(label, '') for label in pred).strip(' |_') for pred in preds]
        true_strings = [''.join(self.data.mapping.get(label, '') for label in true).strip(' |_') for true in trues]

        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(true_string)
            for pred_string, true_string in zip(pred_strings, true_strings)
        ]
        if verbose:
            sorted_ind = np.argsort(char_accuracies)
            print("\nLeast accurate predictions:")
            for ind in sorted_ind[:5]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nMost accurate predictions:")
            for ind in sorted_ind[-5:]:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
            print("\nRandom predictions:")
            random_ind = np.random.randint(0, len(char_accuracies), 5)
            for ind in random_ind:  # pylint: disable=not-an-iterable
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
        mean_accuracy = np.mean(char_accuracies)
        
        if was_training:
            self.network.train()

        return mean_accuracy

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        output_length = self.data.output_shape[0]
        # softmax_output_fn = K.function(
        #     [self.network.get_layer('image').input, K.learning_phase()],
        #     [self.network.get_layer('softmax_output').output]
        # )
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        # Get the prediction and confidence using softmax_output_fn, passing the right input into it.
        # Your code below (Lab 3)
        input_image = np.expand_dims(image, 0)
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()

            input_image = torch.from_numpy(input_image).to(device)
            y_pred, input_lengths = self.network(input_image) # y_pred (T,N,C)

        # input_length = np.array([softmax_output.shape[1]])
        # decoded, log_prob = K.ctc_decode(softmax_output, input_length, greedy=True)
        pred_idx = ctc_decode(y_pred.permute((1,0,2)), input_lengths, output_length) # input requires (N,T,C)
        pred_raw = pred_idx[0] # the batch only contains 1 element

        # pred_raw = K.eval(decoded[0])[0]
        pred = ''.join(self.data.mapping[label] for label in pred_raw).strip(' |_')

        # neg_sum_logit = K.eval(log_prob)[0][0]
        max_logit, _ = torch.max(y_pred.squeeze(dim=1), dim=1)
        # TODO: is this the right confidence?
        conf = np.exp(max_logit.sum())
        # conf = np.exp(-neg_sum_logit)
        
        # Your code above (Lab 3)
        if was_training:
            self.network.train()

        return pred, conf
