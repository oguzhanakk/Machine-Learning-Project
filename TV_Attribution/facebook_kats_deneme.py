#import pandas as pd
#import numpy
#import seaborn as sns
#import matplotlib.pyplot as plt
from tv_attribution import sample


nebiye = sample.read_data('setur-new.csv')

sample.build_model(nebiye,'2022-06-16')