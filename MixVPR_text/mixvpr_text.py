import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn
from sentence_transformers import SentenceTransformer
from models import helper
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModel
import numpy as np

class VPRModel_text(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False,
                text_encoder_name='BAAI/bge-large-en-v1.5',
                embeds_dim=1024,
                is_freeze_vpr=True,
                is_freeze_text=True,
                fusion_type='mlp',
                is_encode_image=False,
                is_encode_text=True,
                is_trainable_text_encoder=True
                 ):
        super().__init__()
        
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        self.is_encode_image = is_encode_image
        self.is_encode_text = is_encode_text
        self.is_trainable_text_encoder = is_trainable_text_encoder

        if is_encode_image:
            self.vpr_encoder = VPRModel(backbone_arch, pretrained, layers_to_freeze, layers_to_crop, agg_arch, agg_config, lr, optimizer, weight_decay, momentum, warmpup_steps, milestones, lr_mult, loss_name, miner_name, miner_margin, faiss_gpu)
            vpr_output_dim = agg_config['out_rows'] * agg_config['out_channels']
            if is_freeze_vpr:
                # Freeze vpr encoder parameters
                for param in self.vpr_encoder.parameters():
                    param.requires_grad = False
        if is_encode_text:
            # self.text_encoder = SentenceTransformer(text_encoder_name)
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)  
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

            if is_freeze_text:
                # Freeze text encoder parameters
                for param in self.text_encoder.parameters():
                    param.requires_grad = False                      
            
            # Define LoRA configuration
            # TaskType.FEATURE_EXTRACTION is appropriate for sentence embedding tasks
            if is_trainable_text_encoder:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=8,  # LoRA rank
                    lora_alpha=32, # LoRA scaling factor
                    lora_dropout=0.1,
                    target_modules=["query", "value"]
                    #target_modules=["query", "key", "value", "base_layer", "dense"]  # specify the modules to apply LoRA
                )

                # Get the PEFT model with LoRA adapters
                self.text_encoder = get_peft_model(self.text_encoder, lora_config)
                # self.text_encoder.add_adapter(lora_config)
        
        text_encoder_dim = 1024        
        self.fusion_type = fusion_type
        self.embeds_dim = embeds_dim        
        
        if is_encode_image and is_encode_text:
            if self.fusion_type == 'transformer':
                self.vpr_adapter = nn.Linear(vpr_output_dim, embeds_dim)  # mix vpr dim embedding
                self.text_adapter = nn.Linear(text_encoder_dim, embeds_dim)  # BGE large has 1024-dim embedding                   
                self.cls = nn.Parameter(torch.randn(1, 1, embeds_dim))
                # Define the core Transformer Encoder stack
                encoder_layer = nn.TransformerEncoderLayer(d_model=embeds_dim, nhead=4, dim_feedforward=4*embeds_dim, batch_first=True)  
                # Input shape: (sequence_length, batch_size, d_model)        
                self.fusion = nn.TransformerEncoder(encoder_layer, 4)
            elif self.fusion_type == 'mlp':
                input_dim = vpr_output_dim + text_encoder_dim
                self.fusion = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, embeds_dim))
                #self.fusion_residual = nn.Linear(input_dim, embeds_dim)
        
        # Call the weight initialization function
        self.apply(self._init_weights)
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # For linear layers, use Kaiming uniform initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # For biases, it's common to initialize them to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        
    # the forward pass of the lightning model
    def forward(self, img, text):
        if self.is_encode_image:
            img_embeds = self.vpr_encoder.backbone(img)
            img_embeds = self.vpr_encoder.aggregator(img_embeds)
        if self.is_encode_text:
            # text_embeds = self.text_encoder.encode(text, normalize_embeddings=True, convert_to_tensor=True)        
            text_tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(img.device)
            model_output = self.text_encoder(**text_tokens)            
            text_embeds = model_output[0][:, 0]
            text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        
        batch_size = img.shape[0]   
        
        if self.is_encode_image and self.is_encode_text:        
            if self.fusion_type == 'transformer':
                img_embeds = self.vpr_adapter(img_embeds)
                text_embeds = self.text_adapter(text_embeds)
                # fusion type is transformer encoder
                # Expand the learnable [CLS] token to match the batch size
                cls_tokens = self.cls.expand(batch_size, -1, -1)                    
                # Prepend the [CLS] token to the input sequence
                # The new sequence length is (original_seq_len + 1)        
                # concat cls_token, image_embeds, text_embeds as input to fusion transformer encoder
                embeds_input = torch.cat([cls_tokens, img_embeds.unsqueeze(dim=1), text_embeds.unsqueeze(dim=1)], dim=1)
                embeds = self.fusion(embeds_input)[:,0,:]
            elif self.fusion_type == 'mlp':
                embeds_input = torch.cat([img_embeds, text_embeds], dim=1)
                embeds = self.fusion(embeds_input) #+ self.fusion_residual(embeds_input)                
        elif self.is_encode_text:
            embeds = text_embeds
        elif self.is_encode_image:
            embeds = img_embeds

        return embeds
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels, texts = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        flat_texts = []
        for i in range(BS):
            for j in range(N):
                flat_texts.append(texts[j][i])

        # Feed forward the batch to the model
        descriptors = self(images, flat_texts) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        # calculate descriptors
        descriptors = self(places, texts)
        return descriptors.detach().cpu()
    
    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                # num_references = val_dataset.dbStruct.numDb
                num_references = val_dataset.num_db
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 50, 100],
                                                gt=positives,
                                                print_results=True,
                                                dataset_name=val_set_name,
                                                faiss_gpu=self.faiss_gpu
                                                )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        return descriptors.detach().cpu()
    
    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 50, 100],
                                                gt=positives,
                                                print_results=True,
                                                dataset_name=val_set_name,
                                                faiss_gpu=self.faiss_gpu
                                                )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
