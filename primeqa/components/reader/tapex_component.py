from primeqa.tableqa.tapex.models.tapex_model import TapexModel    
class Reader():
    
    def __init__(self,runnable_class_name,path_to_config_json=None):
        # if path_to_config_json is None:
        #     path_to_config_json = os.path.realpath(__file__)+runnable_class_name+"_config.json"
        self._class_object = eval(runnable_class_name)(path_to_config_json)
    
    def predict(self, data, queries):
        answers = self._class_object.predict(data,queries)
        return answers
    
    def train(self):
        trained_results = self._class_object.train()
        return trained_results

    def eval(self):
        eval_results = self._class_object.eval()
        return eval_results

