from src.pipeline.pipeline import Pipeline
from src.stages.operation.preprocessor import Preprocessor
from src.stages.source.scrapper import Scrapper

if __name__ == '__main__':
    pipe = Pipeline()
    pipe.add(Scrapper("scrapper_config"))
    pipe.add(Preprocessor("preprocessor_config"))

    pipe.process()
