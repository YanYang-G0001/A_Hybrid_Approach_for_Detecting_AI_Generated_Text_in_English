"""
Model definitions for MGT detection with DeBERTa + Perplexity features.

This module defines:
1. MGTDatasetWithPerplexity: PyTorch Dataset for text classification with perplexity scores
2. DebertaV2WithPerplexity: DeBERTa model extended with perplexity feature fusion
3. Helper functions for model initialization and weight transfer

Reference: https://github.com/Advacheck-OU/ai-detector-coling2025
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoConfig, DebertaV2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    ContextPooler
)


class MGTDatasetWithPerplexity(Dataset):
    """
    PyTorch Dataset for Machine-Generated Text (MGT) detection with perplexity features.
    
    Args:
        data_list (list): List of dictionaries containing text samples and metadata
        perplexity_scores (list): List of perplexity scores corresponding to each sample
        tokenizer: HuggingFace tokenizer for text encoding
        max_length (int): Maximum sequence length for tokenization
    """
    
    def __init__(self, data_list, perplexity_scores, tokenizer, max_length=512):
        self.data_list = data_list
        self.perplexity_scores = perplexity_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(data_list) == len(perplexity_scores), \
            "Data list and perplexity scores must have the same length"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'perplexity_feature': torch.tensor([self.perplexity_scores[idx]], dtype=torch.float32),
            'labels': torch.tensor(item['label'], dtype=torch.long),
            'source': item['source'],
            'sub_source': item['sub_source'],
            'model': item['model']
        }


class DebertaV2WithPerplexity(DebertaV2PreTrainedModel):
    """
    DeBERTa model with perplexity feature fusion for binary classification.
    
    This model extends the standard DeBERTa architecture by concatenating
    a perplexity feature to the pooled output before classification.
    
    Architecture:
        - DeBERTa encoder (768-dim)
        - Context pooler
        - Perplexity fusion (768+1 = 769-dim)
        - MLP classifier: 769 -> 512 -> 256 -> 2
    
    Args:
        config (DebertaV2Config): Model configuration
        hidden_dim1 (int): First hidden layer size in classifier (default: 512)
        hidden_dim2 (int): Second hidden layer size in classifier (default: 256)
        classifier_dropout (float): Dropout rate in classifier (default: 0.5)
    """

    def __init__(self, config, hidden_dim1=512, hidden_dim2=256, classifier_dropout=0.5):
        super().__init__(config)
        self.num_labels = 2  # binary classification
        self.config = config

        # DeBERTa core components
        self.deberta = DebertaV2Model(config)
        
        # Dropout layer
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)

        # Context pooler
        self.pooler = ContextPooler(config)

        # Classification head: input = hidden_size + 1 (perplexity feature)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 1, hidden_dim1),  # 768+1 → 512
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),              # 512 → 256
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim2, self.num_labels)           # 256 → 2
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        perplexity_feature=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass with optional perplexity feature fusion.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs
            attention_mask (torch.FloatTensor): Attention mask
            token_type_ids (torch.LongTensor): Token type IDs
            perplexity_feature (torch.FloatTensor): Perplexity scores [batch_size, 1]
            labels (torch.LongTensor): Ground truth labels for loss calculation
            output_attentions (bool): Whether to output attention weights
            output_hidden_states (bool): Whether to output hidden states
            return_dict (bool): Whether to return ModelOutput object
            
        Returns:
            SequenceClassifierOutput: Model outputs with loss, logits, and optional states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass through DeBERTa encoder
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Pool and dropout
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)

        # Concatenate perplexity feature
        if perplexity_feature is not None:
            # perplexity_feature: [batch_size, 1]
            pooled_output = torch.cat([pooled_output, perplexity_feature], dim=-1)  # [B, 769]
        else:
            # If not provided, use zero padding
            batch_size = pooled_output.size(0)
            zero_feature = torch.zeros(batch_size, 1, device=pooled_output.device)
            pooled_output = torch.cat([pooled_output, zero_feature], dim=-1)

        # Classification
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def initialize_classifier_weights(model, config):
    """
    Initialize the classifier head with Xavier initialization for text features
    and zero initialization for the perplexity feature dimension.
    
    This ensures that initially the model relies primarily on DeBERTa features,
    and the perplexity feature's contribution is learned gradually during training.
    
    Args:
        model (DebertaV2WithPerplexity): The model to initialize
        config (DebertaV2Config): Model configuration
    """
    with torch.no_grad():
        first_layer = model.classifier[0]  # Linear(769, 512)
        # Xavier initialization for text features (first 768 dimensions)
        nn.init.xavier_uniform_(first_layer.weight[:, :config.hidden_size])
        # Zero initialization for perplexity feature (last dimension)
        first_layer.weight[:, config.hidden_size:] = 0.0
        # Zero bias initialization
        nn.init.zeros_(first_layer.bias)


def transfer_pretrained_weights(pretrained_model, new_model):
    """
    Transfer weights from a pretrained DeBERTa model to the new model.
    
    Only transfers the DeBERTa encoder and pooler weights. The classifier
    head is initialized separately since it has a different input dimension.
    
    Args:
        pretrained_model (DebertaV2ForSequenceClassification): Source model
        new_model (DebertaV2WithPerplexity): Target model
    """
    new_model.deberta.load_state_dict(pretrained_model.deberta.state_dict())
    new_model.pooler.load_state_dict(pretrained_model.pooler.state_dict())


def load_pretrained_components(model_name="OU-Advacheck/deberta-v3-base-daigenc-mgt1a"):
    """
    Load pretrained tokenizer, config, and model from HuggingFace Hub.
    
    Args:
        model_name (str): Name or path of the pretrained model
        
    Returns:
        tuple: (tokenizer, config, pretrained_model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    pretrained_model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
    
    print(f" Loaded pretrained components from: {model_name}")
    return tokenizer, config, pretrained_model


def create_model_with_perplexity(
    pretrained_model_name="OU-Advacheck/deberta-v3-base-daigenc-mgt1a",
    hidden_dim1=512,
    hidden_dim2=256,
    classifier_dropout=0.5
):
    """
    Create and initialize a DeBERTa model with perplexity feature fusion.
    
    This is the main function for model creation. It:
    1. Loads pretrained components (tokenizer, config, pretrained model)
    2. Creates a new model with extended classifier
    3. Transfers pretrained weights (encoder + pooler)
    4. Initializes the classifier head
    
    Args:
        pretrained_model_name (str): HuggingFace model name
        hidden_dim1 (int): First hidden layer dimension in classifier
        hidden_dim2 (int): Second hidden layer dimension in classifier
        classifier_dropout (float): Dropout rate in classifier
        
    Returns:
        tuple: (model, tokenizer, config)
    """
    # Load pretrained components
    tokenizer, config, pretrained_model = load_pretrained_components(pretrained_model_name)
    
    # Create new model
    model = DebertaV2WithPerplexity(
        config, 
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        classifier_dropout=classifier_dropout
    )
    
    # Transfer weights from pretrained model
    transfer_pretrained_weights(pretrained_model, model)
    print("Transferred DeBERTa and pooler weights")
    
    # Initialize classifier head
    initialize_classifier_weights(model, config)
    print("Initialized classifier head (perplexity weights = 0)")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready - Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    return model, tokenizer, config


# ============================================================================
# Main execution (for testing/debugging)
# ============================================================================

if __name__ == "__main__":
    # Create model with default settings
    model, tokenizer, config = create_model_with_perplexity()
    
    print("\n" + "="*70)
    print("Model successfully created and initialized!")
    print("="*70)