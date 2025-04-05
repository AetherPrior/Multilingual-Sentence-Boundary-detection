import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from huggingface_hub import HfApi, upload_file, upload_folder


class CustomXLMRClassifier(nn.Module):
    """
    Custom classifier that uses XLM-RoBERTa as the base model and adds two linear layers
    followed by a binary classifier layer that matches the provided checkpoint structure.
    """

    def __init__(self, pretrained_model_name=None, config=None, hidden_size=768,
                 intermediate_size=1568, dropout_rate=0.1, num_labels=1,
                 vocab_size=250002, max_position_embeddings=514, type_vocab_size=1):
        """
        Initialize the model.

        Args:
            pretrained_model_name (str): Name of the pretrained XLM-R model to use
            config (XLMRobertaConfig or None): Configuration for the model
            hidden_size (int): Size of the hidden layer from XLM-R (default is 768)
            intermediate_size (int): Size of the intermediate linear layer
            dropout_rate (float): Dropout probability
            num_labels (int): Number of output labels (2 for binary classification)
            vocab_size (int): Size of vocabulary for embeddings (default is 250002)
            max_position_embeddings (int): Maximum sequence length (default is 514)
            type_vocab_size (int): Size of token type vocabulary (default is 1)
        """
        super(CustomXLMRClassifier, self).__init__()

        # Load the pretrained XLM-R model or create with custom config
        if pretrained_model_name:
            self.base = XLMRobertaModel.from_pretrained(pretrained_model_name)
        else:
            if config is None:
                # Create a custom configuration with specified embedding sizes
                config = XLMRobertaConfig(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=max_position_embeddings,
                    type_vocab_size=type_vocab_size
                )
            self.base = XLMRobertaModel(config)

        # Define the classifier with two linear layers and a binary classifier
        self.dropout = nn.Dropout(dropout_rate)

        # Classifier with three components to match the checkpoint structure
        self.classifier = nn.ModuleDict({
            'linear1': nn.Linear(hidden_size, intermediate_size),
            'linear2': nn.Linear(intermediate_size, 4),
            'binary_classifier': nn.Linear(intermediate_size, num_labels)
        })

        # Activation function
        self.activation = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights for the linear layers."""
        nn.init.xavier_uniform_(self.classifier.linear1.weight)
        nn.init.xavier_uniform_(self.classifier.linear2.weight)
        nn.init.xavier_uniform_(self.classifier.binary_classifier.weight)
        nn.init.zeros_(self.classifier.linear1.bias)
        nn.init.zeros_(self.classifier.linear2.bias)
        nn.init.zeros_(self.classifier.binary_classifier.bias)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, labels=None, **kwargs):
        """
        Forward pass for the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            token_type_ids: Token type IDs (not used by XLM-R but included for compatibility)
            position_ids: Position IDs
            labels: Ground truth labels for loss calculation
            **kwargs: Additional keyword arguments

        Returns:
            SequenceClassifierOutput with loss, logits, and hidden states if specified
        """
        # Get the output from the base XLM-R model
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        # Get the pooled output (using the pooler from XLM-R)
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # First linear layer
        hidden = self.classifier.linear1(pooled_output)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Second linear layer
        hidden = self.classifier.linear2(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Binary classifier
        logits = self.classifier.binary_classifier(hidden)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


# Function to load the model from a checkpoint
def load_from_checkpoint(checkpoint_path, optimizer=None, strict=False):
    """
    Load a CustomXLMRClassifier and optimizer from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        strict (bool): Whether to strictly enforce that the keys in state_dict match
                      the keys returned by this module's state_dict() function

    Returns:
        tuple: (model, optimizer, extra_data) - Loaded model, optimizer (if provided), and
               any additional data from the checkpoint such as epoch, steps, etc.
    """
    # Load the checkpoint to extract model dimensions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Extract embedding dimensions from the state dict
    vocab_size = state_dict['base.embeddings.word_embeddings.weight'].shape[0]
    max_position_embeddings = state_dict['base.embeddings.position_embeddings.weight'].shape[0]
    type_vocab_size = state_dict['base.embeddings.token_type_embeddings.weight'].shape[0]
    hidden_size = state_dict['base.embeddings.word_embeddings.weight'].shape[1]

    # Check if classifier dimensions can be extracted
    if 'classifier.linear1.weight' in state_dict:
        intermediate_size = state_dict['classifier.linear1.weight'].shape[0]
    else:
        intermediate_size = 256  # Default value

    # Create a config that matches the checkpoint dimensions
    config = XLMRobertaConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size
    )

    # Initialize the model with the correct dimensions
    model = CustomXLMRClassifier(
        pretrained_model_name=None,
        config=config,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size
    )

    # Load the state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Structured checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        optimizer_args_radam =[
                {'params': model.base.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}
            ]
        optimizer = torch.optim.RAdam(optimizer_args_radam)
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to the right device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        # Extract any additional information
        extra_data = {k: v for k, v in checkpoint.items()
                     if k not in ['model_state_dict', 'optimizer_state_dict']}
    else:
        # Simple model state dict
        model.load_state_dict(checkpoint, strict=strict)
        extra_data = {}

    # Move model to the right device
    model = model.to(device)

    return model, optimizer, extra_data


# Function to push model to Hugging Face Hub
def push_to_hub(model, tokenizer=None, config=None, checkpoint=None, repo_id=None,
                commit_message="Upload model",
                private=False,
                use_auth_token=None):
    """
    Push model, tokenizer, configuration and checkpoint to the Hugging Face Hub.

    Args:
        model: The model to push
        tokenizer: Optional tokenizer to push
        config: Optional model configuration to push
        checkpoint: Optional checkpoint with optimizer state and training metadata
        repo_id: The name of the repository to push to in format "username/modelname"
        commit_message: Message to commit with
        private: Whether the repository should be private
        use_auth_token: Authentication token (or True to use the cached token)

    Returns:
        URL of the repository
    """
    # Initialize Hub API
    api = HfApi()

    if repo_id is None:
        raise ValueError("repo_id must be specified, e.g., 'username/model-name'")

    # Create the repository if it doesn't exist yet
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            token=use_auth_token
        )
    except Exception as e:
        print(f"Error creating repository: {e}")
        return None

    # Create a temporary directory to store files
    import tempfile
    import os
    import shutil
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the model in PyTorch format
        if checkpoint is not None:
            # Save the full checkpoint including optimizer state and metadata
            torch.save(checkpoint, os.path.join(tmpdirname, "checkpoint.pt"))
        else:
            # Save just the model state dict
            torch.save(model.state_dict(), os.path.join(tmpdirname, "pytorch_model.bin"))

        # Save the model configuration
        if config is None and hasattr(model, 'config'):
            config = model.config

        if config is not None:
            if hasattr(config, 'to_json_file'):
                config.to_json_file(os.path.join(tmpdirname, "config.json"))
            else:
                # If it's a dictionary, save as JSON
                import json
                with open(os.path.join(tmpdirname, "config.json"), 'w') as f:
                    json.dump(config, f)

        # Save the tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(tmpdirname)

        # Upload all files to the Hub
        upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            commit_message=commit_message,
            token=use_auth_token
        )

        # Create a model card if it doesn't exist
        try:
            readme_path = os.path.join(tmpdirname, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write(f"# {repo_id.split('/')[-1]}\n\n")
                    f.write("This model is a fine-tuned version of XLM-RoBERTa with a custom binary classifier.\n\n")
                    f.write("## Model Description\n")
                    f.write("- **Model Architecture:** XLM-RoBERTa with two linear layers and a binary classifier\n")
                    if checkpoint is not None:
                        if isinstance(checkpoint, dict) and 'config' in checkpoint:
                            f.write(f"- **Configuration:** {checkpoint['config']}\n")
                    f.write("- **Framework:** PyTorch\n")

                # Upload README separately
                upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=use_auth_token,
                    commit_message="Add model card"
                )
        except Exception as e:
            print(f"Error creating README: {e}")

    return repo_url


# Example usage
def example_usage():
    # Initialize the model for training
    model = CustomXLMRClassifier(pretrained_model_name="xlm-roberta-base")

    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # For inference and continued training from a checkpoint
    # model, optimizer, extra_data = load_from_checkpoint(
    #     "path/to/checkpoint.pt",
    #     optimizer=optimizer
    # )
    #
    # # Retrieve training information if needed
    # start_epoch = extra_data.get('epoch', 0)
    # global_step = extra_data.get('global_step', 0)
    # print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Example of saving a checkpoint with optimizer state
    # current_epoch = 2
    # global_step = 1000
    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'epoch': current_epoch,
    #     'global_step': global_step,
    #     'config': {
    #         'hidden_size': 768,
    #         'intermediate_size': 256,
    #         'dropout_rate': 0.1,
    #         'num_labels': 2
    #     }
    # }
    # torch.save(checkpoint, 'checkpoint.pt')

    # Option 1: Using the Hugging Face Trainer for pushing to Hub
    from transformers import Trainer, TrainingArguments

    # Example training arguments with Hub integration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        # Hub-specific arguments
        push_to_hub=True,
        hub_model_id="username/my-xlmr-model",  # Replace with your username
        hub_strategy="every_save",  # or "end"
        hub_token="hf_your_token"  # or set HF_TOKEN env variable
    )

    # # Initialize Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     # train_dataset=train_dataset,  # Uncomment and provide your datasets
    #     # eval_dataset=eval_dataset,
    #     # optimizers=(optimizer, None),  # Use your pre-configured optimizer
    # )

    # Train and push to Hub automatically
    # trainer.train(resume_from_checkpoint="path/to/checkpoint.pt")
    # trainer.push_to_hub()

    # Option 2: Manual pushing to Hub with custom checkpoint
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    #
    # # Push the model, tokenizer, and full checkpoint
    # push_to_hub(
    #     model=model,
    #     tokenizer=tokenizer,
    #     checkpoint=checkpoint,  # This includes optimizer state and metadata
    #     repo_id="username/my-xlmr-model",  # Replace with your username
    #     use_auth_token="hf_your_token"  # or set HF_TOKEN env variable
    # )

if __name__ == '__main__':
    model, optimizer, extra_data = load_from_checkpoint('xlm-roberta-base-epoch-2.pth')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': extra_data['epoch'],
        'metrics': extra_data['metrics'],
        'config': {
            'hidden_size': 768,
            'intermediate_size': 1568,
            'dropout_rate': 0.1,
            'num_labels': 2
        }
    }
    from transformers import AutoTokenizer
    push_to_hub(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
        checkpoint=checkpoint,
        repo_id='AetherPrior/mlm-punct',
        use_auth_token=True
        )
