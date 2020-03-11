from src.pipeline.stage import Stage


class Pipeline:

    def process(self, source: Stage, operation: Stage):
        source.run()
        operation.run()
        # execution.run()
