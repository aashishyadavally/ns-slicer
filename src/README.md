# ``NS-Slicer``: A Learning-Based Approach to Static Program Slicing

## Contents

* [Configuration](#configuration)
* [Usage Instructions](#usage-instructions)
  - [For Experiments Replication](#for-experiments-replication)
  - [For Custom Dataset](#for-custom-dataset)

### Configuration

The main entry-point script is ``ns-slicer/src/run.py``. It has the following:

1. Required arguments:

    | Argument         | Description |
    | :--------------: | :---- |
    | ``--data_dir``   | Path to directory containing datasets |
    | ``--output_dir`` | Path to save model outputs |


2. Experiment arguments:

    | Argument                | Default                      | Description |
    | :---------------------: | :--------------------------: | :---- |
    | ``--model_key``         | microsoft/graphcodebert-base | Huggingface pre-trained model key  |
    | ``--pretrain``          |   False                      | Freeze pre-trained model layers |
    | ``--save_predictions``  |   False                      | Cache model predictions during evaluation |
    | ``--use_statement_ids`` |   False                      | Use statement ids in input embeddings |
    | ``--pooling_strategy``  |   mean                       | Pooling strategy |
    | ``--pct``               |  0.15                        | Percentage of code to strip to simulate partial code |
    | ``--load_model_path``   |  None                        | Path to trained model: Should contain the .bin files |
    | ``--do_train``          |  False                       | Whether to run training |
    | ``--do_eval``           |  False                       | Whether to run eval on the dev set |
    | ``--do_eval_partial``   |  False                       | Whether to run eval on the partial snippets from dev set |
    | ``--do_eval_aliasing``  |  False                       | Whether to run variable aliasing on dev set |
    | ``--do_predict``        |  False                       | Whether to predict on given dataset |

3. Optional arguments:

    | Argument               | Default  | Description |
    | :--------------------: | :------: | :---- |
    | ``--max_tokens``       |   512    | Maximum number of tokens in a code example |
    | ``--train_batch_size`` |   64     | Batch size per GPU/CPU for training |
    | ``--eval_batch_size``  |   64     | Batch size per GPU/CPU for evaluation |
    | ``--learning_rate``    |  1e-4    | The initial learning rate for Adam optimizer |
    | ``--weight_decay``     |  0.0     | Weight decay for Adam optimizer |
    | ``--adam_epsilon``     |  1e-8    | Epsilon for Adam optimizer |
    | ``--num_train_epochs`` |  5       | Total number of training epochs to perform |
    | ``--seed``             |  42      | Random seed for initialization |

### Usage Instructions

Follow these instructions:

#### For Experiments Replication

1. For intrinsic evaluation on complete code
   * NS-Slicer with CodeBERT (*off-the-shelf*)
     - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key microsoft/codebert-base --pretrain --do_train
       ```
     - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/codebert-pt/Epoch_4/model.ckpt --model_key microsoft/codebert-base --pretrain --do_eval
       ```
       
   * NS-Slicer with GraphCodeBERT (*off-the-shelf*)
     - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key microsoft/graphcodebert-base --pretrain --do_train
       ```
     - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-pt/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --pretrain --do_eval
       ```
       
   * NS-Slicer with CodeBERT
     - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key microsoft/codebert-base --do_train
       ```
     - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/codebert-ft/Epoch_4/model.ckpt --model_key microsoft/codebert-base --do_eval
       ```
       
   * NS-Slicer with GraphCodeBERT
     - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key microsoft/graphcodebert-base --do_train
       ```
     - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-ft/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_eval
       ```
   
3. For intrinsic evaluation on partial code (NS-Slicer with GraphCodeBERT)
   * Omitting 5% of the statements at both start and end
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-ft/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_eval_partial --pct 0.05
       ```
   * Omitting 10% of the statements at both start and end
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-ft/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_eval_partial --pct 0.10
       ```
   * Omitting 15% of the statements at both start and end
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-ft/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_eval_partial --pct 0.15
       ```
  
4. Ablation study
   * NS-Slicer, *w/o* source code pre-training
      - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key roberta-base --do_train
       ```
      - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/roberta-base/Epoch_4/model.ckpt --model_key roberta-base --do_eval
       ```
         
   * NS-Slicer, *w/o* data-flow pre-training
      - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key microsoft/codebert-base-mlm --do_train
       ```
      - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/codebert-base-mlm/Epoch_4/model.ckpt --model_key microsoft/codebert-base-mlm --do_eval
       ```

   * NS-Slicer, *w/o* mean-pooling
         - Training:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --model_key microsoft/graphcodebert-base --do_train
       ```
      - Inference:
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-max/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_eval
       ```

5. Variable aliasing
   * NS-Slicer with CodeBERT
      ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/codebert-ft/Epoch_4/model.ckpt --model_key microsoft/codebert-base --do_eval_aliasing
      ```
   * NS-Slicer with GraphCodeBERT
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-ft/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_eval_aliasing
       ```

6. For extrsinsic evaluation (i.e., vulnerability detection)
   * NS-Slicer with GraphCodeBERT (off-the-shelf)
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-pt/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_predict
       ```
   * NS-Slicer with GraphCodeBERT
       ```bash
       $ python run.py --data_dir ../data --output_dir ../models --load_model_path ../models/graphcodebert-ft/Epoch_4/model.ckpt --model_key microsoft/graphcodebert-base --do_predict
       ```

#### For Custom Dataset
