import json
from urllib import response
from flask import Flask, render_template, request, redirect, jsonify, send_from_directory
import os
import urllib.request
from werkzeug.utils import secure_filename
import numpy as np
from fileinput import filename
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import os
import pandas
import io
from ast import literal_eval
import time
from flask_cors import CORS
import string
import random

UPLOAD_FOLDER = 'files'

app = Flask(__name__)
CORS(app=app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True

def generate_dataset(dataFile, outputFile):

    print("GENERATING DATASET")

    factor=4
    fil1 = open(dataFile)
    fil3 =  open(outputFile,'w')
    lines1 = fil1.readlines()

    j=0


    count=0
    coords=[]
    for line in lines1:
        if(line.rstrip()==''):
            continue
        elif(line.startswith("-2") or line.startswith('#C: Res= -2')):
            continue   
        elif(line.startswith('#C:')):
            equ = line.split('=')
            coords = equ[len(equ)-1].split(' ')[1:4]
            continue
    
    
        if(line.startswith("-1")):
            fil3.write(str(int(int(coords[0])/factor))+","+str(int(int(coords[1])/factor))+","+str(int(int(coords[2].rstrip())/factor))+','+line)
            continue    

        elif(line.startswith('#Base') or line.startswith('#Steps')):
            continue    

        elif(line.startswith("#Voxel")):
            fil3.write(line.split()[2]+","+line.split()[3]+","+line.split()[4])
            fil3.write('\n')
            continue
        elif(line.startswith("#dmax")):
            continue    

        li = line.split(',');
        flag=0;
        #print(len(labelArray1))
        for i in li:
            if(flag==0):
                label = "0"
                fil3.write(str(int(int(coords[0])/factor))+","+str(int(int(coords[1])/factor))+","+str(int(int(coords[2].rstrip())/factor))+',')
                fil3.write(label)
                flag=1;
            else:
                fil3.write(',');
                fil3.write(i);
        count+=1

def run_model(inp_file):

    print("RUNNING MODEL")

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 

    #convert labels to onehot
    def convertOneHot_test(data):
        y=np.array([int(i[0]) for i in data])
        y_onehot=[0]*len(y)
        for i,j in enumerate(y):
            y_onehot[i]=[0]*(y.max() + 1)
            y_onehot[i][j]=1
        return (y,y_onehot)

    #tensorflow graph definitions
    def weight_variable(shape,name):
        W1 = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(W1,name=name)

    def bias_variable(shape,name):
        b1 = tf.constant(0.1, shape=shape)
        return tf.Variable(b1,name=name)

    def conv3d(x, W):
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def max_pool_2x2x2(x):
        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')
    def avg_pool_2x2x2(x):
        return tf.nn.avg_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')  


    def read_csv(recordsFile):
        writefilename = os.path.join("./", recordsFile + '.tfrecords')
        data = pandas.read_csv("./"+recordsFile).values
        train_data_sample = list(data[0:])
        labelarr = np.array([int(i[0]) for i in train_data_sample])
        x_train = np.array([ i[1:len(i)] for i in train_data_sample])    
        writer = tf.python_io.TFRecordWriter(writefilename)
        train_size=np.shape(labelarr)[0]
        idx=range(train_size)
        for i in range(len(train_idx)):
            feature = x_train[i].tostring()
            if counter%1000==0:
                print(str(counter)+"th iteration")
            features = {
                'myfeatures' : bytes_feature(feature),
                'label' : int64_feature(int(labelarr[i]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            counter+=1
        writer.close()
        return val_size,val_size2

    #data read and pre-processing
    def read_csv2(x,y,recordsFile):
        writefilename = os.path.join("./", recordsFile + '2.tfrecords')
        np_x = np.array(x)
        labelarr = np.array(y)
        mincount = min(np.count_nonzero(labelarr==0),np.count_nonzero(labelarr==1),np.count_nonzero(labelarr==2))
        idx0 = np.where(labelarr==0)
        idx1 = np.where(labelarr==1)
        idx2 = np.where(labelarr==2)
        idx3 = np.where(labelarr==3)
        idx0_new = random.sample(np.ndarray.tolist(idx0[0]),mincount)
        idx1_new = random.sample(np.ndarray.tolist(idx1[0]),mincount)	
        idx2_new = random.sample(np.ndarray.tolist(idx2[0]),mincount)	
        idx3_new = random.sample(np.ndarray.tolist(idx3[0]),mincount)
        labelarr_new = np.hstack([labelarr[idx0_new],labelarr[idx1_new],labelarr[idx2_new],labelarr[idx3_new]])
        x_train1 = np.array(np.reshape(np_x,(-1,3*3*3*4)))
        x_train = np.vstack([x_train1[idx0_new,:],x_train1[idx1_new,:], x_train1[idx2_new,:],x_train1[idx3_new,:]])
        temp = np.array([labelarr_new])
        totArr = np.hstack([temp.T,x_train])
        np.random.shuffle(totArr)
        labelarr_new = totArr[:,0]
        x_train = totArr[:,1:np.shape(totArr)[1]]
        writer = tf.python_io.TFRecordWriter(writefilename)
        for i in range(len(labelarr_new)):
            feature = x_train[i].tostring()
            if counter%100==0:
                print(str(counter)+"th iteration")
            features = {
                'myfeatures' : bytes_feature(feature),
                'label' : int64_feature(int(labelarr_new[i]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            counter+=1
        writer.close()


    def read_and_decode(filename_queue,feature_size):
        reader = tf.TFRecordReader()
        _, serializedStr = reader.read(filename_queue)
        feature_to_type = {
            'myfeatures' : tf.FixedLenFeature([],tf.string),
            'label': tf.FixedLenFeature([], tf.int64)		
        }
        data = tf.parse_single_example(serializedStr, feature_to_type)
        features1 = tf.decode_raw(data['myfeatures'],tf.float64)
        features1.set_shape([feature_size])

        label = tf.cast(data['label'],tf.int32)
        features1 = tf.cast(features1,tf.float32)

        return features1,label

    def read_records(filename,batch_size,num_epochs):
        readfile = os.path.join("./", filename + '.tfrecords')
        filename_queue = tf.train.string_input_producer(
        [readfile], num_epochs=num_epochs)

        features2,label = read_and_decode(filename_queue,1331)
        min_after_dequeue = 100
        capacity = min_after_dequeue + 2 * batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
        [features2, label], batch_size=batch_size, num_threads=2, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
        return example_batch,label_batch	


    def read_records2(filename,batch_size,num_epochs):
        readfile = os.path.join("./", filename + '2.tfrecords')
        filename_queue2 = tf.train.string_input_producer(
        [readfile], num_epochs=num_epochs)

        features2,label2 = read_and_decode(filename_queue2,3*3*3*4)
        min_after_dequeue = 100
        capacity = min_after_dequeue + 2 * batch_size
        example_batch2, label_batch2 = tf.train.shuffle_batch(
        [features2, label2], batch_size=batch_size, num_threads=2, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
        return example_batch2,label_batch2

    def display_help(custom_error):
        """ Displays usage help. """
        print("ERROR - " + custom_error + ".")
        print("HELP:\n\tPARAMETERS:")
        print("\t\tInput file: Path to input file that defines the path for the dataset.")
        print("\t\t--prefix, Output file name prefix (optional): File name prefix for output files. Useful to include the path or to differentiate between different executions using the same filenames as input. Default: outputP1_<mrc filename> and outputP2_<mrc filename>.")
        print("\tEXAMPLE: python Emap2sec.py myinputfile.txt --prefix outputfilename")

    def remove_file_silent(file):
        """
        Tries to remove a file.
        If an exception is thrown (if access is denied of files does not exist), it does not show such error.
        """
        try:
            os.remove(file)
        except:
            pass
    
    y_test_arr = []
    x_test_arr1 = []
    test_data = []
    test_data_1 =[]
    test_data_11 =[]
    reshaped = []
    reshaped2 = []
    fCoords=[]
    fCoords_tr=[]
    x_test_arr2 = []
    y_test_arr2 = []
    test_data2 = []

    filnames=[]
    filnames2=[]
    # Output filename (if None, default is used)
    output_prefix = None


    # Generating filenames for phase 1 and 2
    first_output_prefix = output_prefix + "outputP1_" if output_prefix else "outputP1_"
    second_output_prefix = output_prefix + "outputP2_" if output_prefix else "outputP2_"

    # Adding prefix for tmp files
    tmp_input_prefix = output_prefix.replace('/', '_').replace('~', '') + '_' if output_prefix else '_'
    if tmp_input_prefix[0] == '_':
        tmp_input_prefix = tmp_input_prefix[1:]
    tmp_prefix = 'data/' + tmp_input_prefix + 'TMP_'

    input_location_file = os.path.join("results") + "/" + inp_file
    print(input_location_file)
    fil2 = open(input_location_file,'r')
    j=0
    input_files = []
    input_files_basename = []
    # for filname in fil2:
        # Saving dataset filenames for info messages and output file name
    # one_line_filename = fil2.replace('\n', '')
    # input_files.append(one_line_filename)
    input_files_basename.append(os.path.basename(inp_file))
    print("INFO : Running Emap2sec Phase1 for dataset", fil2)

    df = pandas.DataFrame([line.rstrip().split(',') for line in fil2])
    if(df.empty):
        print("~~~ERROR : Not running Emap2sec for file : %s because input dataset is empty~~~~~",fil2)
    pre_td = pandas.read_csv(io.StringIO(u""+df.to_csv(index=False))).values

    if(np.shape(pre_td)[1] < 1331):
        print("~~~ERROR : Not running Emap2sec for file : %s because input dataset has an error~~~~~",fil2)
    filnames2.append(fil2.name)
        
    test_data2.append(pre_td)
    reshaped2.append((int(test_data2[j][0][0]),int(test_data2[j][0][1]),int(test_data2[j][0][2])))

    test_tmp = []
    coords=[]
    for i in test_data2[j][1:len(test_data2[j])]:
        if i[3] == -1:
            i[3]=3
        test_tmp.append(i[3:])
        coords.append(i[0:3])

    fCoords.append(coords)

    x_test_arr2.append(np.array([ i[1:len(i)] for i in test_tmp]))
    if(np.shape(x_test_arr2[j])[0] == 0):
        j+=1
    y_testi2,_ = convertOneHot_test(test_tmp)
    y_test_arr2.append(y_testi2)
    j+=1

    test_file_count2 = j
    fil2.close()
    filXs = open(tmp_prefix+'XFiles','w')

    filXs.write(str(filnames2))
    filXs.close()

    tf.reset_default_graph()

    lambdA = tf.placeholder(tf.float32)
    beta = tf.placeholder(tf.float32)


    sess2 = tf.Session()
    saver1 = tf.train.import_meta_graph('models/emap2sec_models_exp1/emap2sec_L1_exp.ckpt-108000.meta')
    saver1.restore(sess2,tf.train.latest_checkpoint('models/emap2sec_models_exp1/'))

    graph = tf.get_default_graph()

    convLayers=5
    fc1Layers=2


    wConv_array=[]
    bConv_array=[]

    for i in range(1,convLayers+1,1):
        w=graph.get_tensor_by_name("W_conv"+str(i)+":0")
        b=graph.get_tensor_by_name("b_conv"+str(i)+":0")
        wConv_array.append(w)
        bConv_array.append(b)
        
    W_array=[]
    b_array=[]
    for i in range(1,fc1Layers+1,1):
        w=graph.get_tensor_by_name("W_fc"+str(i)+":0")
        b=graph.get_tensor_by_name("b_fc"+str(i)+":0")
        W_array.append(w)
        b_array.append(b)

    W_fcf = graph.get_tensor_by_name("W_fc5:0")
    b_fcf = graph.get_tensor_by_name("b_fc5:0")

    x_test = tf.placeholder(tf.float32, shape=[None, 1331])
    y_test = tf.placeholder(tf.int64, shape=[None, 1])

    x_test = tf.cast(x_test,tf.float32)
    x_image_test = tf.reshape(x_test, [-1,11,11,11,1])
    h_array1=[]
    h_array2=[]

    for i in range(convLayers):
        if i==0:
            h=tf.nn.relu(conv3d(x_image_test, wConv_array[i]) + bConv_array[i])
        else:
            h=tf.nn.relu(conv3d(h_array1[i-1], wConv_array[i]) + bConv_array[i])
        h_array1.append(h)
    h_pool1 = max_pool_2x2x2(h_array1[i])
    h_pool3_flat = tf.reshape(h_pool1, [-1, 6*6*6*128])

    for i in range(fc1Layers):
        if i==0:
            h=tf.nn.relu(tf.matmul(h_pool3_flat, W_array[i]) + b_array[i])
        else:
            h=tf.nn.relu(tf.matmul(h_array2[i-1], W_array[i]) + b_array[i])
        h_array2.append(h)
    y_conv_test=tf.matmul(h_array2[i], W_fcf) + b_fcf
    y_pred = tf.argmax(y_conv_test,1)

    x_test_arr2_full = []
    y_test_arr2_full = []
    y_test_arr2_full_old = []
    print(input_files_basename)
    for j in range(test_file_count2):
        y_test_arr_reshape = np.reshape(y_test_arr2[j],(len(y_test_arr2[j]),1))	
        y_prob = []
        fil1 = open("results/" + first_output_prefix+input_files_basename[j],'w')
        fil1.write("#"+filnames2[j]+"\n")

        for i in range(len(y_test_arr_reshape)):
            fil1.write(str(y_test_arr_reshape[i][0]))
            fil1.write(';')
        fil1.write('\n')
        indProb=[]
        sInd=0
        pred_labels=[]
        cum_pred_labels=[]
        for ind in range(0,len(y_test_arr2[j]),100):
            sInd = min(len(y_test_arr2[j]),ind+100)
            temProb = sess2.run(tf.nn.softmax(y_conv_test),feed_dict={x_test:x_test_arr2[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})
            if(indProb==[]):
                indProb=temProb
            else:
                indProb = np.ndarray.tolist(np.vstack([np.array(indProb),temProb]))
            pred_labels = y_pred.eval(session=sess2,feed_dict={x_test:x_test_arr2[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})
            for i in range(len(pred_labels)):
                cum_pred_labels.append(pred_labels[i])
                fil1.write(str(pred_labels[i]))
                fil1.write(';')

        fil1.write("\n")

        for probarr in indProb:
            fil1.write(str(probarr)+"\n")
                
                
        fil1.close()
        print("INFO : Wrote the output of Phase1 to "+first_output_prefix+input_files_basename[j])
        y_prob.append(indProb)
        #print("confusion_matrix:")
        #print(sklearn.metrics.confusion_matrix(y_test_arr_reshape, cum_pred_labels)[0:3,0:3])

        print("INFO : Running Emap2sec Phase2 for dataset")

        index=0
        y_prob_reshape = np.zeros((reshaped2[j][2],reshaped2[j][1],reshaped2[j][0],3))
        y_prob_tlabel_reshape2 = np.zeros((reshaped2[j][2],reshaped2[j][1],reshaped2[j][0]))
        y_prob_tlabel_reshape2.fill(3)
        y_prob_truelabel_reshape2 = np.zeros((reshaped2[j][2],reshaped2[j][1],reshaped2[j][0]))
        y_prob_truelabel_reshape2.fill(3)
        idx_coords=fCoords[j]
        idx_coords=[list(it) for it in idx_coords]
        conut=0
        for k in idx_coords:
            if(y_test_arr_reshape[index][0]!=3):
                y_prob_tlabel_reshape2[int(k[2]),int(k[1]),int(k[0])]=cum_pred_labels[index]
            else:
                y_prob_tlabel_reshape2[int(k[2]),int(k[1]),int(k[0])]=y_test_arr_reshape[index]
            y_prob_reshape[int(k[2]),int(k[1]),int(k[0])]=y_prob[0][index]
            y_prob_truelabel_reshape2[int(k[2]),int(k[1]),int(k[0])]=y_test_arr_reshape[index]
            if(y_prob_truelabel_reshape2[int(k[2]),int(k[1]),int(k[0])]==1):
                conut+=1
            
            index+=1
        x_test2 = []
        y_test_2 = []
        y_test_true = []
        for a in range(reshaped2[j][2]):
            for b in range(reshaped2[j][1]):
                for c in range(reshaped2[j][0]):
                    line=[]
                    bkg_count=0
                    if [c,b,a] not in idx_coords:
                        continue
                    for aa in range(-1,2,1):
                        for bb in range(-1,2,1):
                            for cc in range(-1,2,1):
                                if(a+aa < 0 or b+bb < 0 or c+cc < 0 or a+aa >= reshaped2[j][2] or b+bb >= reshaped2[j][1] or c+cc >= reshaped2[j][0]):
                                    line.append([0,0,0,1])
                                    bkg_count+=1
                                elif(np.array_equal(y_prob_reshape[a+aa][b+bb][c+cc],np.array([0,0,0]))):
                                    line.append([0,0,0,1])
                                    bkg_count+=1
                                else:
                                    modi_y_prob = y_prob_reshape[a+aa][b+bb][c+cc]
                                    line.append(np.ndarray.tolist(np.hstack([modi_y_prob,[0]])))
                    if bkg_count == 27:
                        continue
                    x_test2.append(line)
                    y_test_2.append(y_prob_tlabel_reshape2[a][b][c])
                    y_test_true.append(y_prob_truelabel_reshape2[a][b][c])
        x_test_arr2_full.append(x_test2)
        y_test_arr2_full.append(y_test_true)
        y_test_arr2_full_old.append(y_test_2)
        
    filX = open(tmp_prefix+'XFile','w')
    filX.write(str(x_test_arr2_full))
    filX.close()

    filY = open(tmp_prefix+'yFile','w')
    filY.write(str(y_test_arr2_full))
    filY.close()

    filX=open(tmp_prefix+"XFile","r")
    s=filX.readline()
    x_test_arr2_full=literal_eval(s)
    filX.close()

    filXs=open(tmp_prefix+"XFiles","r")
    s=filXs.readline()
    filnames2=literal_eval(s)
    filXs.close()
    test_file_count2=len(filnames2)

    filX=open(tmp_prefix+"yFile","r")
    s=filX.readline()
    y_test_arr2_full=literal_eval(s)
    filX.close()
                        
    tf.reset_default_graph()                     
    sess3 = tf.Session()
    saver = tf.train.import_meta_graph('models/emap2sec_models_exp2/emap2sec_L2_exp.ckpt-20000.meta')
    saver.restore(sess3,tf.train.latest_checkpoint('models/emap2sec_models_exp2/'))

    graph = tf.get_default_graph()

    x_image_test2 = tf.placeholder(tf.float32, shape=[None, 108])
    y_test2 = tf.placeholder(tf.int64, shape=[None, 1])

    x_image_test2 = tf.cast(x_image_test2,tf.float32)
    x_image_test2 = tf.reshape(x_image_test2, [-1,3*3*3*4])
                        
    fc2Layers=5                      
    W_array2=[]
    b_array2=[]
    for i in range(1,fc2Layers+1,1):
        w=graph.get_tensor_by_name("W_fc2"+str(i)+":0")
        b=graph.get_tensor_by_name("b_fc2"+str(i)+":0")
        W_array2.append(w)
        b_array2.append(b)

            
    h_array22=[]
    for i in range(fc2Layers):
        if i==0:
            h=tf.nn.relu(tf.matmul(x_image_test2, W_array2[i]) + b_array2[i])
        else:
            h=tf.nn.relu(tf.matmul(h_array22[i-1], W_array2[i]) + b_array2[i])
        h_array22.append(h)
    W_fcf = graph.get_tensor_by_name("W_fc26:0")
    b_fcf = graph.get_tensor_by_name("b_fc26:0")

                        
                        
    y_conv_test2=tf.matmul(h_array22[i], W_fcf) + b_fcf
    test_correct_prediction2 = tf.equal(tf.argmax(y_conv_test2,1), y_test2[0])
    test_accuracy2 = tf.reduce_mean(tf.cast(test_correct_prediction2, tf.float32))

    y_pred2 = tf.argmax(y_conv_test2,1)                       
    for j in range(test_file_count2):
        y_test_arr_reshape = np.reshape(y_test_arr2_full[j],(len(y_test_arr2_full[j]),1))	
        y_test_arr_reshape_old = y_test_arr2_full_old[j]	
        y_prob2 = []
        y_prob2.append(sess3.run(tf.nn.softmax(y_conv_test2),feed_dict={x_image_test2:np.reshape(x_test_arr2_full[j],(np.shape(x_test_arr2_full[j])[0],3*3*3*4)),y_test2:y_test_arr_reshape}))
        pred_labels2 = y_pred2.eval(session=sess3,feed_dict={x_image_test2:np.reshape(x_test_arr2_full[j],(np.shape(x_test_arr2_full[j])[0],3*3*3*4)),y_test2:y_test_arr_reshape})
        fil1 = open("results/" + second_output_prefix+input_files_basename[j],'w')
        fil1.write("#"+filnames2[j]+"\n")
        conut=0
        for i in range(len(y_test_arr_reshape)):
            if(y_test_arr_reshape[i][0]==1):
                conut+=1
            fil1.write(str(y_test_arr_reshape[i][0]))
            fil1.write(';')
        fil1.write('\n')
        for i in range(len(pred_labels2)):
            if((int(y_test_arr_reshape_old[i])==2 and int(pred_labels2[i])==0) or (int(y_test_arr_reshape_old[i])==1 and int(pred_labels2[i])==0)):
                pred_labels2[i]=y_test_arr_reshape_old[i]
            fil1.write(str(pred_labels2[i]))
            fil1.write(';')

        fil1.write("\n")

        for probarr in y_prob2[0]:
            fil1.write(str(np.ndarray.tolist(probarr))+"\n")

        
        fil1.close()
        print("INFO : Wrote the output of Phase2 to "+second_output_prefix+input_files_basename[j])

    # Deleting tmp files
    remove_file_silent(tmp_prefix+'XFiles')
    remove_file_silent(tmp_prefix+'XFile')
    remove_file_silent(tmp_prefix+'yFile')

# def allowed_file(filename):
# 	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_visual_files(trim, inp1, inp2):
    print(trim)
    print(inp1)
    os.popen("Visual/Visual.pl {} {} -p > {}".format(trim, inp1, os.path.join("static") + "/" + inp1.split("/")[1] + "1.txt"))

    print("RAN FOR INP 1")

    os.popen("Visual/Visual.pl {} {} -p > {}".format(trim, inp2, os.path.join("static") + "/" + inp2.split("/")[1] + "2.txt"))

    print("RAN FOR INP2")

    temp = []

    temp.append("http://127.0.0.1:8000/" + os.path.join("static") + "/" + inp1.split("/")[1] + "1.txt")
    temp.append("http://127.0.0.1:8000/" + os.path.join("static") + "/" + inp2.split("/")[1] + "2.txt")

    print(temp)

    return temp

def rand_string_generator():
    # printing letters
    letters = string.ascii_letters
    return ( ''.join(random.choice(letters) for i in range(20)) )

@app.route('/file-upload', methods=['POST'])
def upload_file():
    print(request.files)
	# check if the post request has the file part
    # print("ds")
    # print(request.files)
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file:
        filename = secure_filename(file.filename)
        new_file_name = rand_string_generator()
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_file_name))
        generate_dataset(os.path.join(app.config['UPLOAD_FOLDER']) + "/" + new_file_name, os.path.join("results") + "/" + new_file_name + ".txt")
        run_model(new_file_name + ".txt")
        visuals = generate_visual_files(os.path.join(app.config['UPLOAD_FOLDER'], new_file_name), os.path.join("results") + "/" + "outputP1_" + new_file_name + ".txt", os.path.join("results") + "/" + "outputP2_" + new_file_name + ".txt")
        # time.sleep(3000)
        # os.remove(os.path.join(app.config['UPLOAD_FOLDER'], new_file_name))
        # os.remove(os.path.join("results") + "/" + new_file_name + ".txt")
        # os.remove(os.path.join("results") + "/" + "outputP1_" + new_file_name + ".txt")
        # os.remove(os.path.join("results") + "/" + "outputP2_" + new_file_name + ".txt")
        resp = jsonify({'visuals' : visuals})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)