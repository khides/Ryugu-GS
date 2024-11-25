from omegaconf import OmegaConf
from sfm.model import Model
from sfm.model_merger import ModelMerger
from logger import Logger
import datetime
from notice import send_notification

def main(logger: Logger, conf: OmegaConf) -> None:
    train_model = Model(
        model_path=conf.train_model_path,
        name=conf.train_model_name,
        logger=logger
        ) # クエリモデルの読み込みs
    train_model.read_model()  
    train_model.generate_camera_poses_json()
    train_model.write_model()




if __name__ == "__main__":
    with open("config.yaml", mode="r") as f:
        conf = OmegaConf.load(f)
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(f"./log/{now}.log")  
    send_notification(
        file = __file__,
        webhook_url=conf.webhook_url,
        method=main ,
        logger = logger,
        conf = conf
        )

