

from TextClassifier import *


test_path = './data/train/'
path = './data/test/'
model = TextClassifier()

data = model.organize_text(path+'pos/',path+'neg/')
tests = model.organize_text(test_path+'pos/',test_path+'neg/')
model.init_tokenizer(data['text'])
X_train  = np.array(model.proccess_text(data['text']))
X_test = np.array(model.proccess_text(tests['text']))
y_train = data['label']
y_test = tests['label']
print(X_train.shape)
model.train(X_train,y_train,X_test,y_test)
scores = model.eval(X_test,y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

import gc; gc.collect()
