from ..cheminf.utils import UnsupportedClassifierError, ModeError, Timer
from ..cheminf.controller import ChemInfController, MODEL_MODES, AUTO_MODES


class ChemInfTimer(Timer):
    def __init__(self):
        super(ChemInfTimer, self).__init__("ChemInf", verbose=1)
        self.start()


class ChemInfOperator(object):
    def __init__(self, controller=None):
        if controller is None:
            self.timer = ChemInfTimer()
            self.controller = ChemInfController()
        else:
            self.controller = controller
        self.mode = self.controller.args.mode
        self.classifier = self.controller.args.classifier

        if self.mode == 'postproc':
            from ..cheminf.postprocessing import PostProcNormal
            self.postproc = PostProcNormal(self.controller)

        elif self.mode == 'preproc':
            from ..cheminf.preprocessing import PreProcNormal
            self.preproc = PreProcNormal(self.controller)

        elif self.mode == 'auto':
            from ..cheminf.preprocessing import PreProcAuto
            self.preproc = PreProcAuto(self.controller)
            if self.controller.config.execute.auto_plus_sum or self.controller.config.execute.auto_plus_plot:
                from ..cheminf.postprocessing import PostProcAuto
                self.postproc = PostProcAuto(self.controller)
            else:
                self.postproc = None

        if self.mode in MODEL_MODES or self.mode in self.controller.auto_modes:
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
                    raise UnsupportedClassifierError(self.controller.args.classifier)

    def __del__(self):
        if hasattr(self, 'timer'):
            self.timer.stop()

        if hasattr(self, 'controller'):
            self.controller.delete()
            if not self.controller.debugging:
                del self

    @staticmethod
    def init_model(controller):
        if controller.args.classifier == 'all':
            from ..cheminf.model import ModelRNDFOR
            from ..cheminf.model import ModelNN
            models = [ModelRNDFOR(controller), ModelNN(controller)]
            return models

        elif controller.args.classifier == 'rndfor':
            from ..cheminf.model import ModelRNDFOR
            model = ModelRNDFOR(controller)

        elif controller.args.classifier == 'nn':
            from ..cheminf.model import ModelNN
            model = ModelNN(controller)

        else:
            raise UnsupportedClassifierError(controller.args.classifier)

        return model

    def get_model(self, classifier=None):
        if classifier is None:
            classifier = self.classifier

        return getattr(self, f"{classifier}_model")

    def get_utils(self, utils=None):
        if utils is None:
            utils = self.mode
        return getattr(self, utils)

    # Todo: Make classifier option 'all' functional
    def run(self):
        if self.classifier:
            if self.classifier == 'all':
                classifiers = self.controller.classifier_types
            else:
                classifiers = [self.classifier]
        else:
            classifiers = [None]

        if self.mode in AUTO_MODES:
            for classifier in classifiers:
                model = self.get_model(classifier)
                model.build()
                model.predict()

        if self.mode == 'postproc' or self.mode in AUTO_MODES:
            self.postproc.run()

        elif self.mode == 'preproc':
            self.preproc.run()

        elif self.mode in MODEL_MODES:
            for classifier in classifiers:
                model = self.get_model(classifier)
                model_activate = model.get(self.mode)
                model_activate()
        else:
            raise ModeError('operator', self.mode)


def run_operator(controller=None):
    cheminf = ChemInfOperator(controller)
    cheminf.run()
