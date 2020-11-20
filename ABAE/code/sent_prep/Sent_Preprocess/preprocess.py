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



#--------------------------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
 
    #------------------------------------------------------------------------------------------------------------------------------------------
    # original mixed dataset
    #(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task(data_dir, aspect2idx)
    #(train_labelled,train_unlabelled,train_aspect_idx)=load_task(data_dir,aspect2idx)
    #(train_labelled,train_unlabelled,aspect_count_tr),(dev_labelled,dev_unlabelled,aspect_count_dv),(test_labelled,test_unlabelled,aspect_count_test)=load_task(data_dir,aspect2idx)


    #------------------------------------------------------------------------------------------------------------------------------------------
    # dataset divided to single and double location entities
    (tr_lab_single_loc,tr_lab_double_loc,tr_unlab_single_loc,tr_unlab_double_loc),(dev_lab_single_loc,dev_lab_double_loc,dev_unlab_single_loc,dev_unlab_double_loc),(tst_lab_single_loc,tst_lab_double_loc,tst_unlab_single_loc,tst_unlab_double_loc)=load_task_loc(data_dir)


    #------------------------------------------------------------------------------------------------------------------------------------------
    # Test set divided to single and double location entities and labels
    (lab_single_loc, lab_double_loc, single_labels, double_labels)=load_task_loc_test(data_dir)


    #--------------------------------------------------------------------------------------------------------------------------------------------
    # divide dataset to single and double location entities
    # single location
    if not os.path.exists('pickled/single/'):
        os.mkdir('pickled/single/')
    with open('pickled/single/sh_flt_train', 'wb') as pfile_train:
        pickle.dump((tr_lab_single_loc,tr_unlab_single_loc),pfile_train)
    pfile_dev = open('pickled/single/sh_flt_dev', 'wb') 
    pickle.dump((dev_lab_single_loc,dev_unlab_single_loc),pfile_dev)
    pfile_dev.close()
    pfile_test = open('pickled/single/sh_flt_test', 'wb') 
    pickle.dump((tst_lab_single_loc,tst_unlab_single_loc),pfile_test)
    pfile_test.close()


    #double location
    if not os.path.exists('pickled/double/'):
        os.mkdir('pickled/double/')
    pfile_train = open('pickled/double/sh_flt_train', 'wb') 
    pickle.dump((tr_lab_double_loc,tr_unlab_double_loc),pfile_train)
    pfile_train.close()
    pfile_dev = open('pickled/double/sh_flt_dev', 'wb') 
    pickle.dump((dev_lab_double_loc,dev_unlab_double_loc),pfile_dev)
    pfile_dev.close()
    pfile_test = open('pickled/double/sh_flt_test', 'wb') 
    pickle.dump((tst_lab_double_loc,tst_unlab_double_loc),pfile_test)
    pfile_test.close()



    #-------------------------------------------------------------------------------------------------------------------------------------------
    # test set processing
    # single location 
    pfile_test = open('pickled/single/sh_flt_testnlabel', 'wb') 
    pickle.dump((lab_single_loc,single_labels),pfile_test)
    pfile_test.close()

    #double location
    pfile_test = open('pickled/double/sh_flt_testnlabel', 'wb') 
    pickle.dump((lab_double_loc,double_labels),pfile_test)
    pfile_test.close()



    #--------------------------------------------------------------------------------------------------------------------------------------------
    """ # original mixed dataset
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

 """

   