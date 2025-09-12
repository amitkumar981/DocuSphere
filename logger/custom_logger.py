#import libraries
import logging
from datetime import datetime
import os
import structlog

class CustomLogger:
    def __init__(self,log_dir='logs'):
        #Ensure the the directory exist
        self.log_dir=os.path.join(os.getcwd(),log_dir)
        os.makedirs(self.log_dir,exist_ok=True)
        
        #create timestamp log file name
        log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path=os.path.join(self.log_dir,log_file)
        
        
    def get_logger(self,name=__file__):
        logger_name=os.path.join(name) 
        #logger=logging.getLogger(logger_name)
        #logger.setLevel(logging.INFO)
        
        #configure file handler
        file_handler=logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        
        #configure console handler
        console_handler=logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[file_handler,console_handler]
        )
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt='iso',utc=True,key='timestamp'),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to='event'),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True
        )
        
        return structlog.get_logger(logger_name)

#if __name__=="__main__":
    #log=CustomLogger()
  #  logger=log.get_logger(__file__)
  #  logger.info("User uploaded a file", user_id=123, filename="report.pdf")
  #  logger.error("Failed to process PDF", error="File not found", user_id=123)
    