import time

import numpy as np

from ..cheminf.controller import ChemInfController


class ChemInfOperator(object):
    def __init__(self):
        self.controller = ChemInfController()
        self.mode = self.controller.args.mode
        self.classifier = self.controller.args.classifier
        if self.mode == 'postproc':
            from ..cheminf.postprocessing import ChemInfPostProc
            self.postproc = ChemInfPostProc(self.controller)
        elif self.mode == 'preproc':
            from ..cheminf.preprocessing import ChemInfPreProc
            self.preproc = ChemInfPreProc(self.controller)
        elif self.mode == 'auto':
            if self.controller.config.execute.auto_plus_resample:
                from ..cheminf.postprocessing import ChemInfPostProc
                self.preproc = ChemInfPostProc(self.controller)
            else:
                self.preproc = None

            if self.controller.config.execute.auto_plus_sum or self.controller.config.execute.auto_plus_plot:
                from ..cheminf.preprocessing import ChemInfPreProc
                self.postproc = ChemInfPreProc(self.controller)
            else:
                self.postproc = None

        if self.mode in self.controller.model_modes or self.mode in self.controller.auto_modes:
            if self.controller.args.classifier == 'all':
                models = self.init_model(self.controller)
                self.rndfor_model = models[0]
                self.nn_model = models[1]
            else:
                if self.controller.args.classifier == 'rndfor':
                    self.rndfor_model = self.init_model(self.controller)
                    self.nn_model = None
                elif self.controller.args.classifier == 'nn':
                    self.nn_model = self.init_model(self.controller)
                    self.rndfor_model = None
                else:
                    raise ValueError("Not a supported classifier")

    @staticmethod
    def init_model(controller):
        if controller.args.classifier == 'all':
            from ..cheminf.models import ModelRNDFOR
            from ..cheminf.models import ModelNN
            models = [ModelRNDFOR(controller), ModelNN(controller)]
            return models
        elif controller.args.classifier == 'rndfor':
            from ..cheminf.models import ModelRNDFOR
            model = ModelRNDFOR(controller)
        elif controller.args.classifier == 'nn':
            from ..cheminf.models import ModelNN
            model = ModelNN(controller)
        else:
            raise ValueError("Wrong classifier input, has to be either 'random_forest' or 'neural_network'")
        return model

    def get_model(self, classifier=None):
        if classifier is None:
            classifier = self.classifier
        return getattr(self, f"{classifier}_model")

    def get_utils(self, utils=None):
        if utils is None:
            utils = self.mode
        return getattr(self, utils)

    # Todo: Debug auto-mode
    # Todo: Make classifier option 'all' functional
    def run(self):
        if self.classifier:
            if self.classifier == 'all':
                classifiers = self.controller.classifier_types
            else:
                classifiers = [self.classifier]
        else:
            classifiers = [None]

        if self.mode == 'auto':
            for classifier in classifiers:
                model = self.get_model(classifier)
                model.build()
                model.predict()
        elif self.mode == 'preproc':
            self.preproc.run()
        elif self.mode == 'postproc' or self.mode in self.controller.auto_modes:
            self.postproc.run()
        elif self.mode in self.controller.model_modes:
            for classifier in classifiers:
                model = self.get_model(classifier)
                model_activate = model.get(self.mode)
                model_activate()
        else:
            raise ValueError("Invalid operation mode")


class Timer(object):
    def __init__(self, func):
        self.args = ChemInfController.get('args')
        self.func = func

    def __call__(self, *args):
        begin = time.time()

        self.func(*args)

        end = time.time()
        run_time = np.round(end - begin, decimals=2)
        print("\nTIME PROFILE\n-----------------------------------")
        print(f"Runtime for {self.func.__name__} was {run_time}s")
        if self.func.__name__ == "multicore":
            print(f"Runtime per core(s) with {self.func.__name__} was {(run_time * self.args.num_core)}s")


@Timer
def run_operator():
    cheminf = ChemInfOperator()
    cheminf.run()


if __name__ == "__main__":
    run_operator()

