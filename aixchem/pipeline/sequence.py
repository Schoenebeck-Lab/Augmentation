
from aixchem.optimization import Optimizer


class PipelineSequencer:
    """
    This class is used to prepare the input for a pipeline.
    
    Essentially, it takes a list as an input.
    The list could look like this:

    input = [
        T(A),  # Step1
        T(B),  # Step2
        [T(C1), T(C2)]  # Parallel steps
        Opt(T, params=[D3, D4], # Optimizer step with different parameters 
        [T(E1),  Opt(T, params=[E2, E3],] # Parallel steps with an optimizer step
    ]

    If something is not an Optimizer, it should be used as is.
    If something is an Optimizer, the Optimizer.grid() method should be called to retrieve the grid of transformers with different parameters.
    
    The goal is to generate a list of sequences that consist of a combination of all possible steps (i.e. execute substeps in parallel).
    For the example input this should look like this:

    sequences = [
        [T(A), T(B), T(C1), T(D3), T(E1)],
        [T(A), T(B), T(C2), T(D3), T(E1)],
        [T(A), T(B), T(C1), T(D4), T(E1)],
        [T(A), T(B), T(C2), T(D4), T(E1)],
        [T(A), T(B), T(C1), T(D3), T(E2)],
        [T(A), T(B), T(C2), T(D3), T(E2)],
        [T(A), T(B), T(C1), T(D4), T(E2)],
        [T(A), T(B), T(C2), T(D4), T(E2)],
        [T(A), T(B), T(C1), T(D3), T(E3)],
        [T(A), T(B), T(C2), T(D3), T(E3)],
        [T(A), T(B), T(C1), T(D4), T(E3)],
        [T(A), T(B), T(C2), T(D4), T(E3)],
    ]

    
    """
    def __init__(self, input_list):

        self.input_list = input_list

        extended = self._extend_optimizers(input_list)

        self.sequences = self._get_sequences(extended)


    def print(self): 
        for seq_id, seq in enumerate(self.sequences):
            print(f"\n[SEQ{seq_id}] {' -> '.join([s.__repr__() for s in seq])}")

    def _extend_optimizers(self, input_list):

        result = []

        for step in input_list:
            
            # Process None steps
            if step is None:
                result.append(None)

            # Process Optimizer steps
            elif isinstance(step, Optimizer):
                result.append([grid_step for grid_step in step.grid()])

            # Process parallel steps
            elif isinstance(step, list):
                
                substeps = []

                for substep in step:
                    if isinstance(substep, Optimizer):
                        substeps.extend([grid_step for grid_step in substep.grid()])
                    else:
                        substeps.append(substep)

                result.append(substeps)

            else:
                # Directly append the step if it's not a list or Optimizer
                result.append(step)

        return result

    
    def _get_sequences(self, input_list):
        # If there are no steps, return a list with an empty sequence
        if not input_list:
            return [[]]
        
        # Split the list of steps into the first step and the rest of the steps
        first, *rest = input_list

        # Recursively generate all sequences of the rest of the steps
        sequences = self._get_sequences(rest)

        # If the first step is a list, generate a sequence for each step in the list
        if isinstance(first, list):
            return [[s] + seq for s in first for seq in sequences]
        # If the first step is not a list, generate a sequence with the first step
        else:
            return [[first] + seq for seq in sequences]



