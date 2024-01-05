'''Class utilities for loading data to PyTorch models.
'''
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, SequentialSampler,
                              RandomSampler, TensorDataset)


def make_dataloader(args, examples, tokenizer, logger, stage, path_to_dataloader=None):
    '''Make ``torch.utils.data.DataLoader`` object from dataset, sampling randomly
    for training dataset split, and sequentially for evaluation dataset splits.

    Arguments:
        args (Namespace): Program arguments.
        examples (list): train/val/test examples.
        tokenizer (transformers.RobertaTokenizer): Word sub-tokenizer.
        logger (logging.getLogger.*): Logger.
        stage (str): One of train/val/test.
        path_to_dataloader (str): Path to save dataloader.

    Returns:
        dataloader (torch.utils.data.DataLoader): Train/val/test data loader.
    '''
    if stage == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size

    try:
        with open(str(path_to_dataloader), 'rb') as handler:
            dataloader = pickle.load(handler)
    except FileNotFoundError:
        (all_input_ids, all_input_masks, all_statement_ids, all_variable_ids,
         all_variable_line_numbers, all_slice_labels) = [], [], [], [], [], []

        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(f"Writing example {ex_index} of {len(examples)}")

            # Truncate to max_stmts.
            variable = example.variable
            newline_idx = tokenizer.encode("\n")[1]
            variable_start, variable_end = example.variable_loc
            input_ids, tokens_variable, tokens_ids, statement_ids, variable_ids = [], [], [], [], []
            for line_id, line in enumerate(example.code.split('\n')):
                if line_id != example.line_number:
                    tokens = tokenizer.tokenize(line)
                    tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                    input_ids += tokens_ids
                else:
                    pre, post = line[: variable_start], line[variable_end:]
                    assert len(pre) + len(variable) + len(post) == len(line)
                    tokens_pre = tokenizer.tokenize(pre)
                    tokens_variable = tokenizer.tokenize(variable)
                    tokens_post = tokenizer.tokenize(post)
                    tokens = tokens_pre + tokens_variable + tokens_post

                    # Extract variable ids.
                    variable_ids = list(range(len(input_ids) + len(tokens_pre),
                                              len(input_ids) + len(tokens_pre) + len(tokens_variable)))
                    assert len(variable_ids) == len(tokens_variable)
                    variable_ids = variable_ids + [-1 for _ in range(args.max_tokens - len(variable_ids))]

                    # Add tokens_ids to input_ids.
                    tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                    input_ids += tokens_ids

                statement_ids += [line_id for _ in range(len(tokens_ids))]

            # Trim to max tokens size.
            if len(input_ids) > args.max_tokens:
                continue

            num_statements = max(statement_ids)
            all_variable_ids.append(variable_ids)

            # Extract labels for slice.
            slice_labels = [0 for _ in range(num_statements + 1)]
            for idx in example.backward_slice:
                slice_labels[idx] = 1
            for idx in example.forward_slice:
                slice_labels[idx] = 1
            slice_labels += [-1 for _ in range(args.max_tokens - len(slice_labels))]
            all_slice_labels.append(slice_labels)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_masks = [1 for _ in range(len(input_ids))]
            # Zero-pad up to the sequence length.
            zero_padding_length = args.max_tokens - len(input_ids)
            input_ids += [tokenizer.pad_token_id for _ in range(zero_padding_length)]
            input_masks += [tokenizer.pad_token_id for _ in range(zero_padding_length)]
            all_input_ids.append(input_ids)
            all_input_masks.append(input_masks)
            # Max-pad up to the number of maximum statements.
            max_padding_length = args.max_tokens - len(statement_ids)
            statement_ids += [args.max_tokens - 1 for _ in range(max_padding_length)]
            all_statement_ids.append(statement_ids)

            all_variable_line_numbers.append(example.line_number)

        dataset = TensorDataset(
                torch.tensor(all_input_ids, dtype=torch.long),
                torch.tensor(all_input_masks, dtype=torch.long),
                torch.tensor(all_statement_ids, dtype=torch.long),
                torch.tensor(all_variable_ids, dtype=torch.long),
                torch.tensor(all_variable_line_numbers, dtype=torch.long),
                torch.tensor(all_slice_labels, dtype=torch.float),
        )

        if stage == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        with open(str(path_to_dataloader), 'wb') as handler:
            pickle.dump(dataloader, handler)

    return dataloader


def make_dataloader_vuldetect(args, examples, tokenizer, logger):
    '''Make ``torch.utils.data.DataLoader`` object from dataset for extrinsic
    task of vulnerability detection.

    Arguments:
        args (Namespace): Program arguments.
        examples (list): train/val/test examples.
        tokenizer (transformers.RobertaTokenizer): Word sub-tokenizer.
        logger (logging.getLogger.*): Logger.

    Returns:
        dataloader (torch.utils.data.DataLoader): Vulnerability detection data loader.
    '''
    batch_size = args.eval_batch_size
    filtered_examples = []

    (all_input_ids, all_input_masks, all_statement_ids, all_variable_ids,
        all_variable_line_numbers) = [], [], [], [], []

    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        # Truncate to max_stmts.
        variable = example.variable
        newline_idx = tokenizer.encode("\n")[1]
        variable_start, variable_end = example.variable_loc
        input_ids, tokens_variable, tokens_ids, statement_ids, variable_ids = [], [], [], [], []
        for line_id, line in enumerate(example.code.split('\n')):
            if line_id != example.line_number:
                tokens = tokenizer.tokenize(line)
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                input_ids += tokens_ids
            else:
                pre, post = line[: variable_start], line[variable_end:]
                assert len(pre) + len(variable) + len(post) == len(line)
                tokens_pre = tokenizer.tokenize(pre)
                tokens_variable = tokenizer.tokenize(variable)
                tokens_post = tokenizer.tokenize(post)
                tokens = tokens_pre + tokens_variable + tokens_post

                # Extract variable ids.
                variable_ids = list(range(len(input_ids) + len(tokens_pre),
                                            len(input_ids) + len(tokens_pre) + len(tokens_variable)))
                assert len(variable_ids) == len(tokens_variable)
                variable_ids = variable_ids + [-1 for _ in range(args.max_tokens - len(variable_ids))]

                # Add tokens_ids to input_ids.
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                input_ids += tokens_ids

            statement_ids += [line_id for _ in range(len(tokens_ids))]

        # Trim to max tokens size.
        if len(input_ids) > args.max_tokens:
            continue

        all_variable_ids.append(variable_ids)

        # The mask has 1 for real tokens and 0 for padding tokens.
        # Only real tokens are attended to.
        input_masks = [1 for _ in range(len(input_ids))]
        # Zero-pad up to the sequence length.
        zero_padding_length = args.max_tokens - len(input_ids)
        input_ids += [tokenizer.pad_token_id for _ in range(zero_padding_length)]
        input_masks += [tokenizer.pad_token_id for _ in range(zero_padding_length)]
        all_input_ids.append(input_ids)
        all_input_masks.append(input_masks)
        # Max-pad up to the number of maximum statements.
        max_padding_length = args.max_tokens - len(statement_ids)
        statement_ids += [args.max_tokens - 1 for _ in range(max_padding_length)]
        all_statement_ids.append(statement_ids)

        all_variable_line_numbers.append(example.line_number)

        filtered_examples.append(example)


    dataset = TensorDataset(
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_input_masks, dtype=torch.long),
            torch.tensor(all_statement_ids, dtype=torch.long),
            torch.tensor(all_variable_ids, dtype=torch.long),
            torch.tensor(all_variable_line_numbers, dtype=torch.long),
    )

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader, filtered_examples
