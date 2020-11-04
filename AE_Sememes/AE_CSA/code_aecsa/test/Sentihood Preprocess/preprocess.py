########################################
#used to parse review text to the form: 'like jeollado like roll sometimes price variety menu'
########################################

from data_utils_sentihood import *
import pickle
import csv

data_dir='../data/sentihood/'
aspect2idx = {
    'general': 0,
    'price': 1,
    'transit-location': 2,
    'safety': 3,
}

#(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task(data_dir, aspect2idx)
#(train_labelled,train_unlabelled,train_aspect_idx)=load_task(data_dir,aspect2idx)
(train_labelled,train_unlabelled,aspect_count_tr),(dev_labelled,dev_unlabelled,aspect_count_dv),(test_labelled,test_unlabelled,aspect_count_test)=load_task(data_dir,aspect2idx)


pfile_train = open('pickled/sh_flt_train', 'ab') 
pickle.dump((train_labelled,train_unlabelled),pfile_train)
pfile_train.close()
pfile_dev = open('pickled/sh_flt_dev', 'ab') 
pickle.dump((dev_labelled,dev_unlabelled),pfile_dev)
pfile_dev.close()
pfile_test = open('pickled/sh_flt_test', 'ab') 
pickle.dump((test_labelled,test_unlabelled),pfile_test)
pfile_test.close()

print('Aspect frequency:\n', aspect_count_tr,aspect_count_dv,aspect_count_test)
print(train_labelled[:3])
print("len(train_labelled) = ", len(train_labelled))
print("len(train_unlabelled) = ", len(train_unlabelled))
print("len(test_labelled) = ", len(test_labelled))
print("len(test_unlabelled) = ", len(test_unlabelled))



#train_labelled.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
#train_unlabelled.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
#val.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
#test.sort(key=lambda x:x[2]+str(x[0])+x[3][0])


""" print(train_aspect_idx)
with open("aspect_idx.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train_aspect_idx) """