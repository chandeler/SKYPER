import logging
from datetime import datetime


now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = logging.getLogger("test")
logger.setLevel(level=logging.INFO)


handler = logging.FileHandler("log/"+now+".log")
handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)


console = logging.StreamHandler()
console.setLevel(logging.INFO)
# console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)




