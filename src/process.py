'''Class utilities for processing data.
'''
import os
import json
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

from tqdm import tqdm


class InputExample:
    '''Data structure for complete/partial code examples.

    Parameters:
        eid (str): Example ID.
        code (str): Complete/partial code example.
        variable (str): Variable for slicing criterion
        variable_loc (tuple): tuple(<start-index, end-index>)
            Start and end indices on line where variable appears.
        line_number (int): Line number for slicing criterion.
        backward_slice (list<int>): List of line numbers indicating backward slice.
        forward_slice (list<int>): List of line numbers indicating forward slice.
    '''
    def __init__(
            self, eid, code, variable, variable_loc, line_number,
            backward_slice=None, forward_slice=None
    ):
        self.eid = eid
        self.code = code
        self.variable = variable
        self.variable_loc = variable_loc
        self.line_number = line_number
        self.backward_slice = backward_slice
        self.forward_slice = forward_slice


class VDInputExample:
    '''Data structure for vulnerability detection code examples.

    Parameters:
        stage (str): train/val/test split.
        cwe (str): CWE ID.
        filename (str): File name containing vulnerability.
        code (str): Extracted code example.
        variable (str): Variable for slicing criterion
        variable_loc (tuple): tuple(<start-index, end-index>)
            Start and end indices on line where variable appears.
        line_number (int): Line number for slicing criterion.
        label (str): "vul"/"non-vul"
    '''
    def __init__(
            self, stage, cwe, filename, code, variable, variable_loc, 
            line_number, label,
    ):
        self.stage = stage
        self.cwe = cwe
        self.filename = filename
        self.code = code
        self.variable = variable
        self.variable_loc = variable_loc
        self.line_number = line_number
        self.label = label


class BaseDataProcessor(ABC):
    '''Base class for processing data.
    '''
    def __init__(self):
        self.path_to_dataset = None

    def get_train_examples(self):
        '''Retrieve examples for train partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"examples_train.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('train')
        return examples

    def get_val_examples(self):
        '''Retrieve examples for validation partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"examples_val.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('val')
        return examples

    def get_test_examples(self):
        '''Retrieve examples for test partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"examples_test.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('test')
        return examples

    @abstractmethod
    def create_examples(self):
        '''Create ``InputExample``-like objects.
        '''
        pass

    def load_examples(self, path_to_file):
        '''Load cached examples.

        Arguments:
            stage (str): One of 'train', 'val', 'test'.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)
        return examples

    def save(self, examples, path_to_file):
        '''Cache examples.

        Arguments:
            examples (list): List of ``InputExample`` objects.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)


class CompleteDataProcessorFromScratch(BaseDataProcessor):
    '''Processes complete code and creates examples.

    Helper utility to build dataset from outputs of data-processing script
    including JavaSlicer.
    '''
    def __init__(self, data_dir='../data'):
        '''Initializes data processor for complete code examples.

        Arguments:
            data_dir (str): Path to datasets.
        '''
        self.path_to_dataset = Path(data_dir)

    def create_examples(self, stage):
        '''Create ``InputExample`` objects.

        Arguments:
            stage (str): One of 'train', 'val', 'test'.
        '''
        projects = sorted(os.listdir(str(self.path_to_dataset)))
        projects = [project for project in projects \
                    if Path(self.path_to_dataset / project).is_dir()]
        num_train = int(0.8 * len(projects))
        num_val = int(0.1 * len(projects))

        if stage == 'train':
            stage_projects = projects[: num_train]
        elif stage == 'val':
            stage_projects = projects[num_train: num_train + num_val]
        elif stage == 'test':
            stage_projects = projects[num_train + num_val: ]

        examples = []
        for project in tqdm(stage_projects):
            json_files = os.listdir(str(self.path_to_dataset / project))
            json_files = [x for x in json_files if x.endswith(".json")]

            for json_file in json_files:
                path_to_file = self.path_to_dataset / project / json_file
                with open(str(path_to_file), 'r') as f:
                    contents = json.load(f)

                file_source_lines = contents['fileSource'].split('\n')
                for method in contents['methods']:
                    try:
                        start_line, end_line = int(method['methodStartLine']), int(method['methodEndLine'])
                        code_lines = file_source_lines[start_line: end_line + 1]
                        code = '\n'.join(code_lines)
                        for slice in method['slices']:
                            variable = slice['variableIdentifier']
                            line_number = int(slice['lineNumber']) - start_line - 1
                            previous_offset = len('\n'.join(file_source_lines[:int(slice['lineNumber']) - 1]))
                            variable_start = int(slice['variableStart']) - previous_offset - 1
                            variable_end = int(slice['variableEnd']) - previous_offset - 1
                            extracted_variable = code_lines[line_number][variable_start: variable_end]
                            if variable != extracted_variable:
                                continue
                            # Unique identifier
                            eid = f"{stage}-{project}-{Path(json_file).stem}-{variable}-{line_number}"
                            # Format slice source.
                            # 1. Remove whitelines.
                            slice_lines = [x.strip() for x in slice['sliceSource'][2:].split('\n') \
                                           if x.strip() != ""]
                            # 2. Remove imports
                            slice_lines = [line for line in slice_lines if not line.startswith('import')]
                            # 3. Remove class definition
                            if 'class' in slice_lines[0]:
                                slice_lines = slice_lines[1:-1]

                            # Extract slice line numbers, backward and forward slices.
                            slice_line_numbers = []
                            line_ctr = 0
                            for slice_line in slice_lines:
                                for _line_number, line in enumerate(code_lines[line_ctr:]):
                                    slice_line_without_space = slice_line.replace(' ', '')
                                    line_without_space = line.replace(' ', '')
                                    if slice_line_without_space == line_without_space:
                                        slice_line_numbers.append(line_ctr + _line_number)
                                        line_ctr = line_ctr + _line_number + 1
                                        break

                            # Skip examples in which there are errors in slice extraction.
                            if len(slice_line_numbers) != len(slice_lines):
                                continue

                            # Extract forward and backward slices.
                            backward_slice = [x for x in slice_line_numbers if x < line_number]
                            forward_slice = [x for x in slice_line_numbers if x > line_number]

                            backward_is_balanced, forward_is_balanced = False, False
                            backward_ratio = len(backward_slice) / (line_number + 1)
                            forward_ratio = len(forward_slice) / (len(code_lines) - line_number)
                            if 0.3 < backward_ratio < 0.7:
                                backward_is_balanced = True

                            if 0.3 < forward_ratio < 0.7:
                                forward_is_balanced = True

                            if not (backward_is_balanced and forward_is_balanced):
                                continue

                            if len(backward_slice) <= 1 or len(forward_slice) <= 1:
                                continue

                            examples.append(
                                InputExample(
                                    eid=eid,
                                    code=code,
                                    variable=variable,
                                    variable_loc=(variable_start, variable_end),
                                    line_number=line_number,
                                    backward_slice=backward_slice,
                                    forward_slice=forward_slice,
                                )
                            )
                    except Exception: pass

        print(f'Number of examples: {len(examples)}')
        path_to_file = self.path_to_dataset / f"examples_{stage}.pkl"
        self.save(examples, path_to_file)
        return examples

    def load_examples(self, path_to_file):
        '''Load cached examples.

        Arguments:
            stage (str): One of 'train', 'val', 'test'.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)
        return examples

    def save(self, examples, path_to_file):
        '''Cache examples.

        Arguments:
            examples (list): List of ``InputExample`` objects.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)


class CompleteDataProcessor(BaseDataProcessor):
    '''Processes complete code and creates examples.

    Helper utility to build dataset from .JSON files.
    '''
    def __init__(self, data_dir='../data'):
        '''Initializes data processor for complete code examples.

        Arguments:
            data_dir (str): Path to datasets.
        '''
        self.path_to_dataset = Path(data_dir)

    def create_examples(self, stage):
        '''Create ``InputExample`` objects.

        Arguments:
            stage (str): One of 'train', 'val', 'test'.
        '''
        path_to_file = self.path_to_dataset / f"{stage}-examples.json"

        with open(str(path_to_file), 'r') as f:
            input_examples = json.load(f)
        
        examples = []
        for ex in tqdm(input_examples):
            examples.append(
                InputExample(
                    eid=ex['eid'],
                    code=ex['code'],
                    variable=ex['variable'],
                    variable_loc=(ex['variable_loc'][0], ex['variable_loc'][1]),
                    line_number=ex['line_number'],
                    backward_slice=ex['backward_slice'],
                    forward_slice=ex['forward_slice'],
                )
            )

        print(f'Number of examples: {len(examples)}')
        path_to_file = self.path_to_dataset / f"examples_{stage}.pkl"
        self.save(examples, path_to_file)
        return examples

    def load_examples(self, path_to_file):
        '''Load cached examples.

        Arguments:
            stage (str): One of 'train', 'val', 'test'.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)
        return examples

    def save(self, examples, path_to_file):
        '''Cache examples.

        Arguments:
            examples (list): List of ``InputExample`` objects.
            path_to_file (pathlib.Path): Destination to cache examples.
        '''
        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)


class PartialDataProcessor(BaseDataProcessor):
    '''Processes complete code to simulate partial code examples.

    Parameters:
        pct (float): Percentage of code to retain.
        path_to_dataset (pathlib.Path): Path to dataset.
    '''
    def __init__(self, pct, data_dir='../data'):
        self.pct = pct
        self.path_to_dataset = Path(data_dir)

    def get_train_examples(self):
        raise ValueError("Training examples not available for PartialDataProcessor.")

    def get_val_examples(self):
        '''Retrieve examples for validation partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"partial_examples_val_{round(self.pct, 2)}.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('val')
        return examples

    def get_test_examples(self):
        '''Retrieve examples for test partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"partial_examples_test_{round(self.pct, 2)}.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('test')
        return examples

    def create_examples(self, stage):
        '''Create ``InputExample`` objects.

        Arguments:
            stage (str): One of 'val', 'test'.
        '''
        # Extract complete data examples.
        path_to_complete_file = Path(self.path_to_dataset) / f"examples_{stage}.pkl"
        complete_examples = self.load_examples(path_to_complete_file)

        examples = []
        for complete_ex in tqdm(complete_examples):
            try:
                code = complete_ex.code
                code_lines = code.split('\n')
                num_lines_to_remove = int(self.pct * len(code_lines))
                if num_lines_to_remove == 0:
                    partial_code_lines = code_lines
                else:
                    partial_code_lines = code_lines[num_lines_to_remove: -num_lines_to_remove]
                partial_code = '\n'.join(partial_code_lines)
                
                line_number = complete_ex.line_number - num_lines_to_remove 
                if line_number < 0 or line_number > len(partial_code_lines):
                    continue

                variable_start, variable_end = complete_ex.variable_loc[0], complete_ex.variable_loc[1]
                assert partial_code_lines[line_number][variable_start: variable_end] == complete_ex.variable

                backward_slice = [x - num_lines_to_remove for x in complete_ex.backward_slice \
                                  if x - num_lines_to_remove >= 0 and x - num_lines_to_remove < len(partial_code_lines)]
                forward_slice = [x - num_lines_to_remove for x in complete_ex.forward_slice \
                                 if x - num_lines_to_remove >= 0 and x - num_lines_to_remove < len(partial_code_lines)]

                if len(backward_slice) <= 1 or len(forward_slice) <= 1:
                    continue

                examples.append(
                    InputExample(
                        eid=complete_ex.eid,
                        code=partial_code,
                        variable=complete_ex.variable,
                        variable_loc=(variable_start, variable_end),
                        line_number=line_number,
                        backward_slice=backward_slice,
                        forward_slice=forward_slice,
                    )
                )
            except Exception: pass

        print(f'Number of examples: {len(examples)}')
        path_to_file = self.path_to_dataset / f"partial_examples_{stage}_{round(self.pct, 2)}.pkl"
        self.save(examples, path_to_file)
        return examples


class AliasingDataProcessor(BaseDataProcessor):
    '''Processes complete code to simulate variable aliasing.
    '''
    def __init__(self, data_dir='../data'):
        '''Initializes data processor for complete code examples.

        Arguments:
            data_dir (str): Path to datasets.
        '''
        self.path_to_dataset = Path(data_dir)


    def get_train_examples(self):
        raise ValueError("Training examples not available for AliasingDataProcessor.")

    def get_val_examples(self):
        '''Retrieve examples for validation partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"aliasing_examples_val.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('val')
        return examples

    def get_test_examples(self):
        '''Retrieve examples for test partition.
        '''
        path_to_file = Path(self.path_to_dataset) / f"aliasing_examples_test.pkl"
        try:
            examples = self.load_examples(path_to_file)
        except FileNotFoundError:
            examples = self.create_examples('test')
        return examples

    def create_examples(self, stage):
        '''Create ``InputExample`` objects.

        Arguments:
            stage (str): One of 'val', 'test'.
        '''
        # Extract complete data examples.
        path_to_complete_file = Path(self.path_to_dataset) / f"examples_{stage}.pkl"
        complete_examples = self.load_examples(path_to_complete_file)

        examples = []
        for complete_ex in tqdm(complete_examples):
            try:
                _, project, filename, _, _ = complete_ex.eid.split('-')
                path_to_json = self.path_to_dataset / project / f"{filename}.json"
                with open(path_to_json, 'r') as f:
                    contents = json.load(f)

                file_source_lines = contents['fileSource'].split('\n')
                for method in contents['methods']:
                    variable_occurrences = {}
                    for slice in method['slices']:
                        if slice['variableIdentifier'] == complete_ex.variable:
                            curr_line_number = int(slice['lineNumber']) - int(method['methodStartLine']) - 1
                            previous_offset = len('\n'.join(file_source_lines[:int(slice['lineNumber']) - 1]))
                            variable_start = int(slice['variableStart']) - previous_offset - 1
                            variable_end = int(slice['variableEnd']) - previous_offset - 1
                            code = complete_ex.code
                            code_lines = code.split('\n')
                            variable_dict = {
                                'variableStart': variable_start,
                                'variableEnd': variable_end,
                            }

                            if curr_line_number not in variable_occurrences:
                                variable_occurrences[curr_line_number] = [variable_dict]
                            else:
                                variable_occurrences[curr_line_number] += [variable_dict]

                relevant_occurences = {k: v for k, v in variable_occurrences.items() if k in complete_ex.forward_slice}
                if len(relevant_occurences) == 0:
                    continue

                aliasing_line = min(list(relevant_occurences.keys()))
                new_code_lines = code_lines[: aliasing_line] + [f"aliasingVar = {complete_ex.variable};"]
                for _line_id, line in enumerate(code_lines[aliasing_line:]):
                    if aliasing_line + _line_id in relevant_occurences.keys():
                        line_occurrences = relevant_occurences[aliasing_line + _line_id]
                        for occ in line_occurrences:
                            occ_start, occ_end = occ['variableStart'], occ['variableEnd']
                            assert line[occ_start: occ_end] == complete_ex.variable
                            new_code_lines.append(line[:occ_start] + "aliasingVar" + line[occ_end:])
                    else:
                        new_code_lines.append(line)
                new_code = '\n'.join(new_code_lines)
                new_forward_slice = [x if x < aliasing_line else x + 1  for x in complete_ex.forward_slice]
                new_forward_slice += [aliasing_line]
                new_forward_slice = sorted(new_forward_slice)

                if len(complete_ex.backward_slice) <= 1 or len(new_forward_slice) <= 1:
                    continue

                examples.append(
                    InputExample(
                        eid=complete_ex.eid,
                        code=new_code,
                        variable=complete_ex.variable,
                        variable_loc=(complete_ex.variable_loc[0], complete_ex.variable_loc[1]),
                        line_number=complete_ex.line_number,
                        backward_slice=complete_ex.backward_slice,
                        forward_slice=new_forward_slice,
                    )
                )
            except: pass

        print(f'Number of examples: {len(examples)}')
        path_to_file = self.path_to_dataset / f"aliasing_examples_{stage}.pkl"
        self.save(examples, path_to_file)
        return examples


class VulDetectDataProcessor:
    '''Processes code examples in vulnerability detection dataset.
    '''
    def __init__(self, data_dir='../dataset'):
        '''Initializes data processor for complete code examples.

        Arguments:
            data_dir (str): Path to datasets.
        '''
        self.path_to_dataset = Path(data_dir)

    def get_examples(self):
        '''Create ``VDInputExample`` objects.
        '''
        with open(str(Path(self.path_to_dataset) / 'filtered_methods.json'), 'r') as f:
            items = json.load(f)

        examples = []
        for stage in ['train', 'val', 'test']:
            for stage_example in items[stage]:
                for method in stage_example['methodArguments']:
                    try:
                        method_source = stage_example['methodSource']
                        method_source_lines = method_source.split('\n')
                        variable = method['variableIdentifier']
                        line_number = int(method['variableLineNumber'])

                        previous_offset = len('\n'.join(method_source_lines[:line_number]))
                        variable_start = int(method['variableStart']) - previous_offset - 1
                        variable_end = int(method['variableEnd']) - previous_offset - 1

                        examples.append(
                            VDInputExample(
                                stage=stage,
                                cwe=stage_example['cwe'],
                                filename=stage_example['fileName'],
                                code=method_source,
                                variable=variable,
                                variable_loc=(variable_start, variable_end),
                                line_number=line_number,
                                label=method['sliceLabel'],
                            )
                        )
                    except: pass

        print(f'Number of examples: {len(examples)}')
        return examples
