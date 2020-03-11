from src.pipeline.stage import Stage


class Pipeline:

    def __init__(self):
        self.properties = {}
        self.stages = []

    def add(self, stage: Stage):
        stage.attach(self)
        self.stages.append(stage)
        self.properties.update(stage.push_attributes())

    def process(self):
        for stage in self.stages:
            stage.run()
