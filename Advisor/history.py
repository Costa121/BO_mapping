from openbox.utils.history import History
from typing import List

class CompressHistory(History):
    def __init__(self, task_id='OpenBox', num_objectives=1, num_constraints=0, config_space=None, ref_point=None, meta_info=None):
        super().__init__(task_id, num_objectives, num_constraints, config_space, ref_point, meta_info)
        
    def update_compression_indices(self, used_indices: List):
        self.used_indices = used_indices