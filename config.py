import copy
import logging

import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
torch.manual_seed(0)
import tensorflow as tf
tf.random.set_seed(0)
import keras
keras.utils.set_random_seed(0)

# evaluation
from deep.model.metrics import compute_metrics
# complexity
from ptflops import get_model_complexity_info


# Base model configuration - general class with training and testing methods
class PyBaseModel:
    def __init__(self, model: nn.Module, param, memory_input_resolution):
        self.model = model
        self.memory_input_resolution = memory_input_resolution

        self.epoch = int(param['epoch'])
        self.lr = param['lr']
        self.patience = param['patience']
        self.min_delta = param['min_delta']

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train_val_loop(self, train_loader, model_type = -1):        
        # considering the possibility of only having training dataset and not evaluation
        # the different types of dataloader are assigned

        if len(train_loader) == 2:
            trainloader, validloader = train_loader
            if len(validloader != 0):
                validation_activated = True
                # Evaluation early stopping definition
                best_val_loss = float('inf')
                epochs_no_improve = 0
            else:
                validation_activated = False
                trainloader = trainloader[0]
        else:
            validation_activated = False
            trainloader = trainloader[0]
    
        epochs_metrics = [] # variable generated to save metrics per epoch
        self.model.train()
        for e in range(self.epoch):
            # reset metric values
            running_metrics = [0.0, 0.0, 0.0, 0.0]
            for maps, labels in trainloader:

                try: 
                    self.optimizer.zero_grad()
                    ps = self.model.forward(maps)
                    loss = self.criterion(ps, labels)
                    running_metrics[0] += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    # Considering that the activation function is a sigmoid is necessary to round the number
                    # to obtain binary output
                    preds = ps.round()
                    running_metrics = compute_metrics(labels, preds, running_metrics)
                
                except:
                    print("Problema con el tamaño del batch")
                    print(maps.shape)

            self.scheduler.step()
            training_loss,training_f1,training_acc,training_gmean = [(running_metrics[0]/len(trainloader)),
                                                                     (running_metrics[1]/len(trainloader)),
                                                                     (running_metrics[2]/len(trainloader)),
                                                                     (running_metrics[3]/len(trainloader))
                                                                    ]
            
            if validation_activated:
                valid_loss, metrics_f1, metrics_acc, metrics_gmean = self.test_performance(validloader)
                if model_type < 0:
                    epochs_metrics.append([e,training_loss,training_f1,training_acc,training_gmean,
                                       valid_loss, metrics_f1, metrics_acc, metrics_gmean])
                
                else:
                    epochs_metrics.append([model_type,e,training_loss,training_f1,training_acc,training_gmean,
                                       valid_loss, metrics_f1, metrics_acc, metrics_gmean])
                    
                # Early stopping check
                if (best_val_loss - valid_loss) > self.min_delta:
                    best_val_loss, epochs_no_improve = [valid_loss, 0]
                    # Milestone, saved the last model with the best params 
                    # Copy the dictionary of model parameters
                    cloned_state_dict = copy.deepcopy(self.model.state_dict())
                    # Detach the cloned state dictionary
                    for key in cloned_state_dict:
                        cloned_state_dict[key] = cloned_state_dict[key].detach().clone()
                else:
                    epochs_no_improve += 1

                # If the model does not improve for the patience number configured in the model class
                # the training loop is stopped 
                if epochs_no_improve >= self.patience:
                    logging.debug(f"Early stopping triggered")
                    break
        
            # If no validation is done, every round the parameters of the model are saved
            else:
                epochs_metrics.append([e,training_loss,training_f1,training_acc,training_gmean])
                # Copy the dictionary of model parameters
                cloned_state_dict = copy.deepcopy(self.model.state_dict())
                # Detach the cloned state dictionary
                for key in cloned_state_dict:
                    cloned_state_dict[key] = cloned_state_dict[key].detach().clone()

            return cloned_state_dict, epochs_metrics

    def test_performance(self, test_loader, loss = True):
        
        if len(test_loader) == 0:
            epoch_loss,epoch_f1,epoch_acc,epoch_gmean = [0,0,0,0]
        else:
            # reset metric values
            running_metrics = [0.0, 0.0, 0.0, 0.0]
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for maps, labels in test_loader:
                    ps = self.model.forward(maps)
                    val_loss = self.criterion(ps, labels)
                    running_metrics[0] += val_loss.item()
                    preds = ps.round()
                    running_metrics = compute_metrics(labels, preds, running_metrics)

            self.model.train()
        
            epoch_loss = running_metrics[0]/len(test_loader)
            epoch_f1 = running_metrics[1]/len(test_loader)
            epoch_acc = running_metrics[2]/len(test_loader)
            epoch_gmean = running_metrics[3]/len(test_loader)

        if loss:
            return [epoch_loss,epoch_f1,epoch_acc,epoch_gmean]
        else:
            return [epoch_f1,epoch_acc,epoch_gmean]

    def memory_info (self):
       
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"El numero de parametros del modelo es: {pytorch_total_params}\n")
        macs, params = get_model_complexity_info(self.model, self.memory_input_resolution, as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))

        return pytorch_total_params, macs

class TfBaseModel:
    def __init__(self, 
                 model: keras.Model, 
                 param, 
                 loss = keras.losses.BinaryCrossentropy(), 
                 optimizer = keras.optimizers.Adam(learning_rate=1e-3),
                 metric = 'accuracy'):
        
        self.model = model
        
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metric = metric

        self.epoch = int(param['epoch'])
        self.patience = param['patience']
        self.min_delta = param['min_delta']
        self.batch_size = param['batch_size']

    def train_val_loop (self, train_loader, model_type = -1):
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=[self.metric]
        )

        callbacks = []
        # considering the possibility of only having training dataset and not evaluation
        # the different types of dataloader are assigned
        if len(train_loader) == 2 and len(train_loader[1][0]) != 0:          
            trainloader, validloader = train_loader
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                min_delta=self.min_delta,
                restore_best_weights=True
                    ))
            
            trainloader = [element.numpy() for element in trainloader]
            trainloader[0] = np.moveaxis(trainloader[0], 1, -1)
            validloader = [element.numpy() for element in validloader]
            validloader[0] = np.moveaxis(validloader[0], 1, -1)
            
            history = self.model.fit(
                trainloader[0], trainloader[1],
                validation_data=(validloader[0], validloader[1]),
                epochs=self.epoch,
                callbacks=callbacks,
                batch_size=self.batch_size,
                verbose = 0
            )

        else:
            trainloader = train_loader[0]
            trainloader = [element.numpy() for element in trainloader]
            trainloader[0] = np.moveaxis(trainloader[0], 1, -1)

            history = self.model.fit(
                trainloader[0], trainloader[1],
                epochs=self.epoch,
                callbacks=callbacks,
                batch_size=self.batch_size,
                verbose = 0
            )
        return self.model, history
        
    def test_performance(self, test_loader):
        loss_fn = self.loss_fn
        running_metrics = [0.0, 0.0, 0.0, 0.0]
        
        for maps, labels in test_loader:
            xtest = maps.detach().numpy() # type: ignore
            # the input to the conv is by default set to "channel_last" and the modification
            # does not work properly. Necessary to transpose the matrix
            xtest = np.moveaxis(xtest, 1, -1)           
            ytest = labels.numpy() # type: ignore
        
            # model inference and metrics computation
            logits = self.model(xtest, training=False)  # Logits for this minibatch
            loss_value = loss_fn(ytest, logits)
            running_metrics[0] += loss_value.numpy()
            try:
                if self.model.final_act.activation == keras.activations.sigmoid:
                    preds = np.round(logits)
                else: 
                    preds = np.argmax(logits, axis=1).astype(np.float32)
            except:
                logging.debug("error detecting final activation function")
            running_metrics = compute_metrics(ytest, preds, running_metrics)
    
        
        epoch_loss = running_metrics[0]/len(test_loader)
        epoch_f1 = running_metrics[1]/len(test_loader)
        epoch_acc = running_metrics[2]/len(test_loader)
        epoch_gmean = running_metrics[3]/len(test_loader)
    
        logging.debug(f"Test --- F1-score: {epoch_f1}, acc: {epoch_acc}, gmean: {epoch_gmean}")
        return [epoch_loss,epoch_f1,epoch_acc,epoch_gmean]


