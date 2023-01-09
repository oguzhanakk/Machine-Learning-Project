from tv_attribution import sample,autogluon

train_data,test_data = autogluon.read_data('setur-new.csv','asdasd')

"""
print(train_data)
print()
print(test_data)
"""

build_model = autogluon.build_model(train_data,test_data,'Value','setur',48)

best_model,params = autogluon.predict(build_model,test_data)