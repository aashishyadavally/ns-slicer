'''Model classes.
'''
import torch
import torch.nn as nn

from transformers import RobertaModel


class SliceMLP(nn.Module):
    '''Backward/Forward slicing classification head.
    '''
    def __init__(self, config):
        super(SliceMLP, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # Second hidden layer
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        # Output layer
        self.fc3 = nn.Linear(config.hidden_size, 1)
        self.forward_activation = torch.nn.GELU()
        # Dropout layer
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None \
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
  
    def forward(self, x):
        # Add first hidden layer
        x = self.forward_activation(self.fc1(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add second hidden layer
        x = self.forward_activation(self.fc2(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add output layer
        x = self.fc3(x)

        outputs = torch.sigmoid(x)
        return outputs


class AutoSlicingModel(nn.Module):
    '''Automated static slicing model which combines a pre-trained language model (PLM)
    with ``SliceMLP`` backward/forward classification heads.

    Parameters:
        max_tokens (int): Maximum number of tokens in a code example.
        use_statement_ids (bool): Use statement ids in input embeddings. If ``True``,
            assign statement IDs to the tokens within statements.
        pooling_strategy (str): Mean/Max pooling strategy.
        roberta (transformers.RobertaModel): Pre-Trained ``RobertaModel`` object.
        back_mlp (SliceMLP): Backward static slicing classification head.
        forward_mlp (SliceMLP): Forward static slicing classification head.
        loss_criterion (torch.nn.*): Binary cross-entropy loss.    
    '''
    def __init__(self, args, config):
        super(AutoSlicingModel, self).__init__()
        self.max_tokens = args.max_tokens
        self.use_statement_ids = args.use_statement_ids
        self.pooling_strategy = args.pooling_strategy
        self.roberta = RobertaModel.from_pretrained(args.model_key, config=config)

        for param in self.roberta.parameters():
            if args.pretrain:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if self.use_statement_ids:
            self.statement_embeddings = nn.Embedding(args.max_tokens, config.hidden_size)

        self.back_mlp = SliceMLP(config)
        self.forward_mlp = SliceMLP(config)
        self.loss_criterion = nn.BCELoss(reduction='mean')

    def forward(self, inputs_ids, inputs_masks, statements_ids, variables_ids,
                variables_line_numbers, slices_labels=None):
        device = inputs_ids.device if inputs_ids is not None else 'cpu'
        inputs_embeddings = self.roberta.embeddings.word_embeddings(inputs_ids)

        if self.use_statement_ids:
            inputs_embeddings += self.statement_embeddings(statements_ids)

        roberta_outputs = self.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=inputs_masks,
            output_attentions = True,
            output_hidden_states = True,
        )

        hidden_states = roberta_outputs.hidden_states
        # Hidden_states has four dimensions: the layer number (for e.g., 13),
        # batch number (for e.g., 8), token number (for e.g., 256),
        # hidden units (for e.g., 768).

        # Choice: First Layer / Last layer / Concatenation of last four layers /
        # Sum of all layers.
        outputs_embeddings = hidden_states[-1]
        batch_preds = {'back': [], 'forward': []}
        batch_true = {'back': [], 'forward': []}
        batch_loss = torch.tensor(0, dtype=torch.float, device=device)

        for _id, output_embeddings in enumerate(outputs_embeddings):
            statements_embeddings = []
            variable_ids = variables_ids[_id][torch.ne(variables_ids[_id], -1)]
            variable_toks_embeddings = output_embeddings[variable_ids]
            if self.pooling_strategy == 'mean':
                variable_embedding = torch.mean(variable_toks_embeddings, dim=0)
            elif self.pooling_strategy == 'max':
                variable_embedding = torch.max(variable_toks_embeddings, dim=0).values

            item_statements_ids = statements_ids[_id][torch.ne(statements_ids[_id], self.max_tokens - 1)]
            num_statements_in_item = torch.max(item_statements_ids).item()
            for sid in range(num_statements_in_item + 1):
                _statement_ids = (item_statements_ids == sid).nonzero().squeeze()
                statement_toks_embeddings = output_embeddings[_statement_ids]
                if self.pooling_strategy == 'mean':
                    statement_embedding = torch.mean(statement_toks_embeddings, dim=0)
                elif self.pooling_strategy == 'max':
                    statement_embedding = torch.max(statement_toks_embeddings, dim=0).values
                statements_embeddings.append(statement_embedding)

            back_statements_embeddings = statements_embeddings[:variables_line_numbers[_id]]
            forward_statements_embeddings = statements_embeddings[variables_line_numbers[_id] + 1:]
            preds = {
                'back': self.back_mlp(
                    torch.stack([torch.cat((x, variable_embedding)) \
                                  for x in back_statements_embeddings])
                ).squeeze(),
                'forward': self.forward_mlp(
                    torch.stack([torch.cat((x, variable_embedding)) \
                                  for x in forward_statements_embeddings])
                ).squeeze(),
            }
            batch_preds['back'].append(preds['back'])
            batch_preds['forward'].append(preds['forward'])
            if slices_labels is not None:
                item_slice_labels = slices_labels[_id][slices_labels[_id] != -1]
                true = {
                    'back': item_slice_labels[:variables_line_numbers[_id]],
                    'forward': item_slice_labels[variables_line_numbers[_id] + 1:],
                }
                batch_true['back'].append(true['back'])
                batch_true['forward'].append(true['forward'])
                back_loss = self.loss_criterion(preds['back'], true['back'])
                forward_loss = self.loss_criterion(preds['forward'], true['forward'])
                batch_loss += back_loss + forward_loss

        if slices_labels is None:
            return batch_preds
        else:
            return batch_loss, batch_preds, batch_true
