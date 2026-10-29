from pydantic import BaseModel
from typing import Optional
class ProcessRequest(BaseModel):
    file_id : str = None
    chunk_size : Optional[int]=100
    overlap_size:Optional[int]=20
    breakpoint_threshold_type : Optional[str] = "percentile"
    breakpoint_threshold_amount:Optional[float]= 95
    do_emantic_chunk:Optional[int]=1
    do_reset:Optional[int]=0
