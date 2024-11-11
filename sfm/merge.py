from omegaconf import OmegaConf
from sfm.model import Model
from sfm.model_merger import ModelMerger
from logger import Logger
import datetime
from notice import send_notification

def merge(logger: Logger, conf: OmegaConf) -> None:
    train_model = Model(
        model_path=conf.train_model_path, 
        name= conf.train_model_name,
        logger=logger
        ) # トレインモデルの読み込み        
    train_model.read_model()

    query_model = Model(
        model_path=conf.query_model_path,
        name=conf.query_model_name,
        logger=logger
        ) # クエリモデルの読み込みs
    query_model.read_model()
    
    merger = ModelMerger(
        query_model=query_model,
        train_model=train_model,
        merge_model_path=conf.merge_model_path,
        merge_model_name=conf.merge_model_name,
        logger=logger
        )
    merger.merge(
        estimate_type=conf.estimate_type,
        show_plot=conf.show_plot, 
        save_plot=conf.save_plot)



if __name__ == "__main__":
    with open("config.yaml", mode="r") as f:
        conf = OmegaConf.load(f)
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(f"./log/{now}.log")  
    send_notification(
        file = __file__,
        webhook_url=conf.webhook_url,
        method=merge ,
        logger = logger,
        conf = conf
        )

