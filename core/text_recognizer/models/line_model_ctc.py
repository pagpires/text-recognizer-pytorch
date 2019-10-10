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
from text_recognizer.networks.line_crnn import line_crnn
from text_recognizer.networks.ctc import ctc_decode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        default_network_args = {'window_width': 14, 'window_stride': 5} # NOTE: this is default for load_weight thus cannot be changed during train
        if network_args is None:
            network_args = {}
        network_args = {**default_network_args, **network_args}
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    # NOTE CTC loss requires blank_idx, input/output lengths, thus need reimplement fit()
    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None, **train_args):
        if callbacks is None:
            callbacks = []
        if train_args.get('pretrained', False):
            self.load_weights()
            print('loaded pretrained network')

        train_sequence = DatasetSequence(
            dataset.x_train, dataset.y_train, 
            batch_size=batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn
        )

        print(f"Total #training: {len(train_sequence.dataset)}")
        print(f"Total #params: {sum([param.nelement() for param in self.network.parameters()])}")

        self.network.to(device)
        self.network.train()

        optimizer = self.optimizer()(self.network.parameters(), lr=3e-4) # RMSProp is better than Adam in this case
        blank_idx = self.data.num_classes-1
        loss_fn = self.loss()(blank=blank_idx, reduction='mean', average_frames=True)
        
        validation_interval = 5
        score = self.evaluate(dataset.x_test, dataset.y_test)
        print(f"Validation score: {score:.4f}")
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_sequence, 0):

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                # NOTE this assumes blank_idx only occurs in continuity in the last part of seq
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

            print(f"[{epoch+1}, {i+1}] loss: {running_loss/(i+1):.5f}")

            if epoch % validation_interval == (validation_interval-1):
                score = self.evaluate(dataset.x_test, dataset.y_test)
                print(f"Validation score: {score:.4f}")
        
        print('Finished Training')

    def loss(self):
        return CTCLoss # torch_baidu_ctc.CTCLoss converges faster than nn.CTCLoss
    
    def optimizer(self):
        return torch.optim.RMSprop

    def evaluate(self, x, y, batch_size: int = 16, verbose: bool = True) -> float:
        blank_idx = self.data.num_classes - 1
        output_length = self.data.output_shape[0]
        test_sequence = DatasetSequence(x, y, batch_size, format_fn=self.batch_format_fn)
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()
            preds_raw = []
            input_lengths = []
            labels_raw = []
            
            running_loss = 0
            for i, batch in enumerate(test_sequence):
                batch_x, batch_y = map(lambda out: out.to(device), batch)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # log_soft_max (T, B, num_classes)
                log_soft_max, batch_input_lengths = map(lambda out: out.to("cpu"), self.network(batch_x))
                preds_raw.append(log_soft_max.permute(1,0,2))
                input_lengths.append(batch_input_lengths)
                labels_raw.append(batch_y.to("cpu"))
                output_lengths = (torch.sum(batch_y != blank_idx, dim=1)).to(torch.long).cpu()
                
                loss = self.loss()(average_frames=True, blank=blank_idx, reduction='mean')(log_soft_max, batch_y.cpu(), batch_input_lengths, output_lengths)
                running_loss += loss.item()
            # preds_raw: (B, T, C)
            preds_raw, input_lengths = torch.cat(preds_raw), torch.cat(input_lengths)
            labels_raw = torch.cat(labels_raw).numpy() # (B, output_length)
        print(f"Validation loss: {running_loss/(i+1):.4f}")
        print((torch.argmax(preds_raw, dim=2)!=79).sum())
        print(torch.argmax(preds_raw, dim=2)[0])
        
        preds = ctc_decode(preds_raw, input_lengths, output_length)

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
            for ind in random_ind:
                print(f'True: {true_strings[ind]}')
                print(f'Pred: {pred_strings[ind]}')
        mean_accuracy = np.mean(char_accuracies)
        
        if was_training:
            self.network.train()

        return mean_accuracy

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        output_length = self.data.output_shape[0]
        
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        input_image = np.expand_dims(image, 0)
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()

            input_image = torch.from_numpy(input_image).to(device)
            y_pred, input_lengths = self.network(input_image) # y_pred (T,N,C)

        pred_idx = ctc_decode(y_pred.permute((1,0,2)), input_lengths, output_length) # arg[0] requires (N,T,C)
        pred_raw = pred_idx[0] # the batch only contains 1 element

        pred = ''.join(self.data.mapping[label] for label in pred_raw).strip(' |_')

        max_logit, _ = torch.max(y_pred.squeeze(dim=1), dim=1)
        # TODO: implement DP to get the right conf for best path
        conf = torch.exp(max_logit.sum())
        
        if was_training:
            self.network.train()

        return pred, conf

class LineModelCRNN(LineModelCtc):
    """Model for recognizing handwritten text in an image of a line, using CTC loss/decoding."""
    def __init__(self,
                 dataset_cls: type = EmnistLinesDataset,
                 network_fn: Callable = line_crnn,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        default_dataset_args: dict = {}
        if dataset_args is None:
            dataset_args = {}
        dataset_args = {**default_dataset_args, **dataset_args}

        if network_args is None:
            network_args = {}
        
        # Model.__init__()
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None
