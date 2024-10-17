import numpy as np
from tensorflow import keras
import small_functions as sf
import data_processing as dp




X,Y = dp.get_sudent_data()
sf.print_2d_data(X)
sf.print_2d_data(Y)