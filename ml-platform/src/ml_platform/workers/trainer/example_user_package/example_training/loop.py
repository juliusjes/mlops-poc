from ml_platform.workers.trainer.plugin_api.context import TrainingContext
from ml_platform.workers.trainer.plugin_api.training_loop import TrainingLoop

#from ml_platform.workers.trainer.builtin.loops. import TrainingResult


class MyLoop(TrainingLoop):

    def train(ctx: TrainingContext):


        return