'''Entry-point to experiments module.

Check README.md for example commands to run all experiments in paper.
'''
import json
import pickle
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

import torch
from torch.optim import AdamW

from transformers import (get_linear_schedule_with_warmup,
                          RobertaTokenizer, RobertaConfig)

from process import (CompleteDataProcessor, PartialDataProcessor,
                     AliasingDataProcessor, VulDetectDataProcessor)
from load import make_dataloader, make_dataloader_vuldetect
from models import AutoSlicingModel
from utils import set_seed, compute_metrics, display_stats


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(dataloader, model, args, epoch_stats=None):
    '''Evaluate model performance.

    Arguments:
        dataloader (torch.utils.data.DataLoader): Val/test data loader.
        model (models.AutoSlicingModel): Trained model.
        args (Namespace): Program arguments.
        epoch_stats (dict): While training, is equivalent to evaluation statistics
            for previous epoch. During inference, is ``None``.

    Returns:
        (tuple): Tuple of evaluation performance statistics, and label pairs.
    '''
    # Tracking variables
    total_eval_loss = 0
    # Evaluate data for one epoch
    label_pairs = {
        'back': {'true': [], 'preds': []},
        'forward': {'true': [], 'preds': []},
    }

    if not epoch_stats:
        epoch_stats = {}

    for batch in tqdm(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        # Tell pytorch not to bother with constructing the compute
        # graph during the forward pass, since this is only needed
        # for backpropogation (training).
        with torch.no_grad():
            batch_loss, batch_preds, batch_true = model(batch[0], batch[1],
                                                        batch[2], batch[3],
                                                        batch[4], batch[5])
            # Accumulate the validation loss.
            total_eval_loss += batch_loss.item()

        # Move labels to CPU
        for slice_type in ['back', 'forward']:
            curr_true = [x.tolist() for x in batch_true[slice_type]]
            label_pairs[slice_type]['true'] += curr_true

            curr_preds = []
            for item_preds in batch_preds[slice_type]:
                curr_preds.append([1 if x > 0.5 else 0 for x in item_preds.tolist()])
            label_pairs[slice_type]['preds'] += curr_preds

    # Calculate the average loss over all of the batches.
    eval_loss = total_eval_loss / len(dataloader)
    eval_metrics = compute_metrics(label_pairs)

    # Record all statistics.
    return {
        **epoch_stats,
        **{'Epoch evaluation loss': eval_loss},
        **eval_metrics,
    }, label_pairs


def predict(dataloader, model, args):
    '''Predict backward/forward slice for custom datasets.
    
    Arguments:
        dataloader (torch.utils.data.DataLoader): Val/test data loader.
        model (models.AutoSlicingModel): Trained model.
        args (Namespace): Program arguments.

    Returns:
        predictions (list): Lists of backward slice, slicing criterion, and
            forward slice, respectively.
    '''
    # Evaluate data for one epoch
    predictions = []
    for batch in tqdm(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        # Tell pytorch not to bother with constructing the compute
        # graph during the forward pass, since this is only needed
        # for backpropogation (training).
        with torch.no_grad():
            batch_preds = model(batch[0], batch[1], batch[2], batch[3], batch[4])

        # Move labels to CPU
        for i, _item_back in enumerate(batch_preds['back']):
            if len(_item_back.size()) == 0:
                item_back = []
            else:
                item_back = [j for j, x in enumerate(_item_back.tolist()) if x > 0.5 ]
            line_number = batch[4][i].item()
            _item_forward = batch_preds['forward'][i]
            if len(_item_forward.size()) == 0:
                item_forward = []
            else:
                item_forward = [line_number + j + 1 for j, x in enumerate(_item_forward.tolist()) if x > 0.5 ]
            predictions.append([item_back, line_number, item_forward])
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required arguments
    parser.add_argument("--data_dir", required=True, type=str, help="Path to datasets directory.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to save model outputs.")

    ## Experiment arguments
    parser.add_argument("--model_key", default='microsoft/graphcodebert-base',
                        type=str, help="Model string.",
                        choices=['microsoft/codebert-base', 'microsoft/graphcodebert-base', 'roberta-base'])
    parser.add_argument("--pretrain", action='store_true',
                        help='Use xBERT model off-the-shelf')
    parser.add_argument("--save_predictions", action='store_true',
                        help='Cache model predictions during evaluation.')
    parser.add_argument("--use_statement_ids", action='store_true',
                        help="Use statement ids in input embeddings")
    parser.add_argument("--pooling_strategy", default='mean',
                        type=str, choices=['mean', 'max'], help="Pooling strategy.")
    parser.add_argument("--pct", default=0.15, type=float,
                        help="Percentage of code to strip to simulate partial code.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_partial", action='store_true',
                        help="Whether to run eval on the partial snippets in dev set.")
    parser.add_argument("--do_eval_aliasing", action='store_true',
                        help="Whether to run variable aliasing on dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to predict on given dataset.")
 
    ## Optional arguments
    parser.add_argument("--max_tokens", default=512, type=int,
                        help="Maximum number of tokens in a code example.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.warning(f"Device: {args.device}, Number of GPU's: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    # Make directory if output_dir does not exist
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_key)
    model = AutoSlicingModel(
                args,
                config=RobertaConfig.from_pretrained(args.model_key)
    )

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path), strict=False)

    model.to(args.device)
    print(model)
    print()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")


    if args.do_train:
        cdp = CompleteDataProcessor()
        logger.info('Loading training data.')
        train_examples = cdp.get_train_examples()
        logger.info('Constructing data loader for training data.')
        train_dataloader = make_dataloader(
            args, train_examples, tokenizer, logger, 'train',
            Path(args.data_dir) / f"dataloader_train.pkl",
        )

        logger.info('Loading validation data.')
        val_examples = cdp.get_val_examples()
        logger.info('Constructing data loader for validation data.')
        val_dataloader = make_dataloader(
            args, val_examples, tokenizer, logger, 'val',
            Path(args.data_dir) / f"dataloader_val.pkl",
        )
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() \
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() \
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          eps=args.adam_epsilon)
        max_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=max_steps*0.1,
                                                    num_training_steps=max_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")

        training_stats = []
        model.zero_grad()

        for epoch in range(args.num_train_epochs):
            training_loss, num_train_steps = 0, 0
            label_pairs = {
                'back': {'true': [], 'preds': []},
                'forward': {'true': [], 'preds': []},
            }

            model.train()
            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                batch_loss, batch_preds, batch_true = model(batch[0], batch[1],
                                                            batch[2], batch[3],
                                                            batch[4], batch[5])
                training_loss += batch_loss.item()
                for slice_type in ['back', 'forward']:
                    curr_true = [x.tolist() for x in batch_true[slice_type]]
                    label_pairs[slice_type]['true'] += curr_true

                    curr_preds = []
                    for item_preds in batch_preds[slice_type]:
                        curr_preds.append([1 if x > 0.5 else 0 for x in item_preds.tolist()])
                    label_pairs[slice_type]['preds'] += curr_preds
                num_train_steps += 1

                batch_loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                # Update the learning rate
                scheduler.step()

            epoch_tr_loss = training_loss / len(train_dataloader)
            epoch_eval_metrics = compute_metrics(label_pairs)
            epoch_back_acc = epoch_eval_metrics['BACK']['Accuracy']
            epoch_forward_acc = epoch_eval_metrics['FORWARD']['Accuracy']

            logger.info(f"Epoch {epoch}, Training loss: {epoch_tr_loss}")
            logger.info(f"Epoch {epoch}, Training accuracy for Backward Slicing: {epoch_back_acc}")
            logger.info(f"Epoch {epoch}, Training accuracy for Forward Slicing: {epoch_forward_acc}")
            
            # After the completion of one training epoch, measure performance
            # on validation set.
            logger.info('Measuring performance on validation set.')
            # Put the model in evaluation mode--the dropout layers behave
            # differently during evaluation.
            model.eval()
            training_stats, _ = evaluate(val_dataloader, model, args,
                                         epoch_stats={
                                            'Epoch training loss': epoch_tr_loss,
                                            'Epoch backward slicing accuracy': epoch_back_acc,
                                            'Epoch forward slicing accuracy': epoch_forward_acc,
                                         }
                                )
            print(training_stats)

            epoch_output_dir = Path(output_dir) / f'Epoch_{epoch}'
            epoch_output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving model to {epoch_output_dir}")
            torch.save(model.state_dict(), str(epoch_output_dir / 'model.ckpt'))


    if args.do_eval:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()
        # Load test data.
        cdp = CompleteDataProcessor()
        logger.info('Loading test data.')
        test_examples = cdp.get_test_examples()
        logger.info('Constructing data loader for testing data.')
        test_dataloader = make_dataloader(
            args, test_examples, tokenizer, logger, 'test',
            Path(args.data_dir) / f"dataloader_test.pkl",
        )

        stats, label_pairs = evaluate(test_dataloader, model, args)
        display_stats(stats)

        if args.save_predictions:
            with open(str(output_dir / 'predictions.pkl'), 'wb') as f:
                pickle.dump(label_pairs, f)


    if args.do_predict:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()
        # Load test data.
        vdp = VulDetectDataProcessor()
        logger.info('Loading test data.')
        examples = vdp.get_examples()

        logger.info('Constructing data loader for testing data.')
        test_dataloader, new_examples = make_dataloader_vuldetect(args, examples, tokenizer, logger)
        predictions = predict(test_dataloader, model, args)
        assert len(new_examples) == len(predictions)

        examples_with_predictions = {'train': [], 'val': [], 'test': []}
        for ex_id, ex in enumerate(new_examples):
            predicted_slice = []
            for lnum, line in enumerate(ex.code.split('\n')):
                if lnum in predictions[ex_id][0] or \
                   lnum == predictions[ex_id][1] or \
                   lnum in predictions[ex_id][2]:
                    predicted_slice.append(line)
            predicted_slice = '\n'.join(predicted_slice)

            examples_with_predictions[ex.stage].append({
                'cwe': ex.cwe,
                'fileName': ex.filename,
                'methodSource': ex.code,
                'variableIdentifier': ex.variable,
                'variableStart': ex.variable_loc[0],
                'variableEnd': ex.variable_loc[1],
                'lineNumber': ex.line_number,
                'predictedBackwardSlice': predictions[ex_id][0],
                'predictedForwardSlice': predictions[ex_id][2],
                'predictedSlice': predicted_slice,
                'label': ex.label,
            })

        with open('../dataset/filtered_methods_with_predictions.json', 'w') as f:
            json.dump(examples_with_predictions, f, indent=2)


    if args.do_eval_partial:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()
        # Load test data.
        pdp = PartialDataProcessor(pct=args.pct)
        logger.info('Loading test data.')
        test_examples = pdp.get_test_examples()
        logger.info('Constructing data loader for testing data.')
        test_dataloader = make_dataloader(
            args, test_examples, tokenizer, logger, 'test',
            Path(args.data_dir) / f"partial_dataloader_test_{1 - args.pct}.pkl",
        )

        stats, label_pairs = evaluate(test_dataloader, model, args)
        display_stats(stats, "Printing evaluation statistics for Partial Dataset:")


    if args.do_eval_aliasing:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()
        # Load test data.
        adp = AliasingDataProcessor()
        logger.info('Loading test data.')
        test_examples = adp.get_test_examples()
        logger.info('Constructing data loader for testing data.')
        test_dataloader = make_dataloader(
            args, test_examples, tokenizer, logger, 'test',
            Path(args.data_dir) / f"aliasing_dataloader_test.pkl",
        )

        stats, label_pairs = evaluate(test_dataloader, model, args)
        display_stats(stats, "Printing evaluation statistics for Aliasing Dataset:")
