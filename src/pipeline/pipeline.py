from src.pipeline.stage import Stage


class Pipeline:

    def process(self, source: Stage):
        source.run()
        # transformer.run()
        # executor.run()
