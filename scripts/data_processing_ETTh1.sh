# For ETTh1 - Save ETTh1.csv in data/ folder
python -c "
from src.utils import load_and_prepare_data; 
load_and_prepare_data('ETTh1.csv', 100, 50, 25, [1,2,3,4,5,6,7], 5)
"