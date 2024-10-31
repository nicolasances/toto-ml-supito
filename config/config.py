
from totoapicontroller.model.TotoConfig import TotoConfig, CloudProvider
from totoapicontroller.model.singleton import singleton

@singleton
class Config(TotoConfig): 
    
    def __init__(self):
        super().__init__(cloud_provider=CloudProvider.AWS)
        
        self.logger.log("INIT", "Configuration loaded!")
        
    def get_api_name(self) -> str:
        return "toto-ml-supito"
        