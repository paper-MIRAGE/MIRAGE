# For ETTh1 - Save ETTh1.csv in data/ folder
python -c "
from src.utils import load_and_prepare_eICU_data; 
load_and_prepare_eICU_data('eICU.h5', 2, 2)
"