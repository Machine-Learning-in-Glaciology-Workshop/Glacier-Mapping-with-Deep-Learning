import nicesnappy.utils as utils
from typing import Dict, List
import abc


class Operator(abc.ABC):
    @abc.abstractmethod
    def apply(self, input=None):
        pass

    def __call__(self, input=None):
        return self.apply(input)


class SNAPOperator(Operator):
    def __init__(self, name: str, parameters: Dict[str, str]=None):
        self.name = name
        self.__init_parameters(parameters)

    def apply(self, input):
        output = utils.snap.GPF.createProduct(self.name, self.parameters, input)
        return output

    def __init_parameters(self, parameters: Dict[str, str]=None):
        self.parameters = utils.snap.HashMap()
        if parameters is not None:
            for parameter_name, parameter_value in parameters.items():
                self.parameters.put(parameter_name, parameter_value)


class Read(Operator):
    def __init__(self, input_path: str):
        self.input_path = input_path

    def apply(self, input=None):
        if input is not None:
            raise ValueError()
        output = utils.snap.ProductIO.readProduct(self.input_path)
        return output


class Write(Operator):
    def __init__(self, output_path: str, format: str):
        self.output_path = output_path
        self.format = format

    def apply(self, input):
        output_file = utils.snap.File(self.output_path)
        incremental = self.__get_incremental()
        progress_monitor = self.__get_progress_monitor()
        utils.snap.GPF.writeProduct(input, output_file, self.format, incremental, progress_monitor)
        return input

    def __get_incremental(self):
        if self.format == "BEAM-DIMAP":
            return True
        return False

    def __get_progress_monitor(self):
        ProgressMonitor = utils.snap.jpy.get_type("com.bc.ceres.core.PrintWriterProgressMonitor")
        JavaSystem = utils.snap.jpy.get_type("java.lang.System")
        progress_monitor = ProgressMonitor(JavaSystem.out)
        return progress_monitor


class WriteFromProductIO(Operator):
    def __init__(self, output_path: str, format: str):
        self.output_path = output_path
        self.format = format

    def apply(self, input):
        utils.snap.ProductIO.writeProduct(input, self.output_path, self.format)
        return input


class Chain(Operator):
    def __init__(self, operators: List[Operator]):
        self.operators = operators

    def apply(self, input=None):
        output = input
        for operator in self.operators:
            output = operator(output)
        return output


class Branches(Operator):
    def __init__(self, branches: List[Operator]):
        self.branches = branches

    def apply(self, input=None):
        output = []
        for operator in self.branches:
            branch_output = operator(input)
            output.append(branch_output)
        return output


class Map(Operator):
    def __init__(self, operator: Operator):
        self.operator = operator

    def apply(self, inputs: List):
        outputs = []
        for input in inputs:
            output = self.operator(input)
            outputs.append(output)
        return outputs


class Zip(Operator):
    def __init__(self, operator: Operator):
        self.operator = operator

    def apply(self, inputs: List[List]):
        outputs = []
        inputs1 = inputs[0]
        inputs2 = inputs[1]
        for input1, input2 in zip(inputs1, inputs2):
            output = self.operator([input1, input2])
            outputs.append(output)
        return outputs


class Flatten(Operator):
    def apply(self, inputs: List[List]):
        outputs = [_ for sublist in inputs for _ in sublist]
        return outputs

    
class GetByIndices(Operator):
    def __init__(self, indices: List[int] or int):
        self.indices = indices

    def apply(self, inputs: List):
        if isinstance(self.indices, int):
            output = inputs[self.indices]
            return output
        outputs = []
        for index in self.indices:
            output = inputs[index]
            outputs.append(output)
        return outputs
