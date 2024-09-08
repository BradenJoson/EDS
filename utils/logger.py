import logging
import datetime
import sys
sys.path.append("../..")
from configs import settings

cfgs = settings.get_settings()

date = str(datetime.date.today())

file_handler = logging.FileHandler("log/%s_log_%s_%s.txt" % (date, cfgs.data.dataset_name, cfgs.msg))
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
global_logger = logging.getLogger('global')
global_logger.addHandler(file_handler)

global_logger.setLevel(logging.INFO)
logger = global_logger



