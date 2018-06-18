import numpy as np


DIR='ans_good/'

file_acc = [
[ "ans_rfc_between_before_after.txt", 0.6107 ] ,
[ "ans_rfc_between.txt", 0.6132 ] ,
[ "ans_rfc_distance.txt", 0.6119 ] ,
[ "ans_rfc_POS.txt", 0.5943 ] ,
[ "ans_rfc.txt", 0.4152 ] ,
[ "ans_svm_all.txt", 0.5122 ] ,
[ "ans_svm_between_before_after_novalid.txt", 0.7665 ] ,
[ "ans_svm_between_before_after.txt", 0.7649 ] ,
[ "ans_svm_between.txt", 0.7425 ] ,
[ "ans_svm_distance.txt", 0.7610 ] ,
[ "ans_svm_POS.txt", 0.7603 ] ,
[ "ans_svm.txt", 0.5722 ] ,
[ "ans_xgb_between_before_after_POS_dis.txt", 0.7179 ] ,
[ "ans_xgb_between_before_after_POS.txt", 0.6799 ] ,
[ "ans_xgb_between_before_after.txt", 0.6833 ] ,
[ "ans_xgb_between.txt", 0.7084 ] ,
[ "ans_xgb.txt", 0.5634 ],
[ "ans_xgb_between_before_after_POS_dis_extreme.txt", 0.7694]
]

label2num = {'Other': 0,
             'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
             'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
             'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
             'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
             'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
             'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
             'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
             'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
             'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

num2label = {0: 'Other',
             1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
             3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
             5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
             7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
             9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
             11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
             13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
             15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
             17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


vote = np.zeros(shape=(2717,19))

for file, acc in file_acc:
    idx = 0
    f = open(DIR + file,"r")
    for line in f.readlines():
        tmp = line.rstrip("\n").split("\t")
        class_name = tmp[1]

        vote[idx,label2num[class_name]] += acc
        idx += 1
    f.close()

vote_result = np.argmax(vote, axis=1)

with open('ensemble.txt','w') as f:
    for idx,ele in enumerate(vote_result):
        f.write('%d\t%s\n' %(idx + 8001, num2label[ele]))