from src.pipeline.stage import Stage


class Pipeline:

    def __init__(self):
        self.properties = {}
        self.stages = []

    """
    Add concrete stage to pipeline. It will also add all stage's properties to the flow.
    """

    def add(self, stage: Stage):
        stage.attach(self)
        self.stages.append(stage)
        self.properties.update(stage.gather_attributes())

    """
    It triggers run() method of added stages.
    """

    def process(self):
        for stage in self.stages:
            stage.run()

    """
    It triggers test() method of added stages.
    """

    def test(self):
        for stage in self.stages:
            stage.test()
