from tkinter import *
window=Tk()




def predict():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import timeit
    from sklearn import metrics

    a=float(e1_value.get())
    b=float(e2_value.get())


    c=e3_value.get()
    
    if c[0]=='A':
        x=0*10+float(c[1])
    elif c[0]=='B':
        x=1*10+float(c[1])
    elif c[0]=='C':
        x=2*10+float(c[1])
    elif c[0]=='D':
        x=3*10+float(c[1])
    elif c[0]=='E':
        x=4*10+float(c[1])
    else:
        x=5*10+float(c[1])
        
    print(x)
    d=e4_value.get()
    if d=='MORTGAGE':
        d=2
    elif d=='RENT':
        d=3
    elif d=='OWN':
        d=1
    elif d=='OTHER':
        d=4
    elif d=='ANY':
        d=6
    else:
        d=5
        

    
    e=e5_value.get()
    if e=='SOURCE VERIFIED':
        e=1
    elif e=='VERIFIED':
        e=2
    else :
        e=3
        
    f=float(e6_value.get())
    
    g=float(e7_value.get())
    h=float(e8_value.get())
    i=float(e9_value.get())
    
    j=e10_value.get()
    if j=='W':
        j=0
    else :
        j=1
    
    
    k=float(e11_value.get())
    l=float(e12_value.get())
    m=e13_value.get()
    if m== 'INDIVIDUAL':
        m=1
    else:
        m=2
        
    n=float(e14_value.get())
    o=float(e15_value.get())
    p=float(e16_value.get())

    
    


    df=pd.read_csv("data/train_indessa.csv")
    del df['funded_amnt'] #correlated to loan amount
    del df['funded_amnt_inv'] #correlated to loan amount
    del df['mths_since_last_record']
    del df['desc'] #not required
    del df['verification_status_joint']
    del df['zip_code'] #not related
    del df['batch_enrolled'] #no impact
    del df['addr_state'] #not related
    del df['mths_since_last_major_derog']
    del df['emp_title'] #since it got many categorical values which is not important so this attribute can be removed from dataset
    del df['title']
    del df['purpose']

    df['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
    df['term'] = pd.to_numeric(df['term'], errors='coerce')

    print('Transform: emp_length...')
    df['emp_length'].replace('n/a', '0', inplace=True)
    df['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
    df['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
    df['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
    df['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
    df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')


    print('Transform: last_week_pay...')
    df['last_week_pay'].replace(to_replace='th week', value='', regex=True, inplace=True)
    df['last_week_pay'].replace(to_replace='NA', value='', regex=True, inplace=True)
    df['last_week_pay'] = pd.to_numeric(df['last_week_pay'], errors='coerce')

    print('Transform: sub_grade...')
    df['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
    df['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)

    df['application_type'].replace(to_replace='INDIVIDUAL', value='1', regex=True, inplace=True)
    df['application_type'].replace(to_replace='JOINT', value='2', regex=True, inplace=True)

    df['initial_list_status'].replace(to_replace='w', value='0', regex=True, inplace=True)
    df['initial_list_status'].replace(to_replace='f', value='1', regex=True, inplace=True)

    del df['grade']
    del df['member_id']
    df=df.loc[0:20000,:]




    df['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
    df['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
    df['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
    df['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
    df['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
    df['sub_grade'] = pd.to_numeric(df['sub_grade'], errors='coerce')

    print('Transform done.')


    '''
    Missing values imputation
    '''
    cols = ['term', 'loan_amnt' , 'last_week_pay', 'int_rate', 'sub_grade', 'annual_inc', 'dti', 'mths_since_last_delinq',  'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']
    for col in cols:
        print('Imputation with Median: %s' % (col))
        df[col].fillna(df[col].median(), inplace=True)

    cols = ['acc_now_delinq', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med','delinq_2yrs','inq_last_6mths','pub_rec']
    for col in cols:
        print('Imputation with Zero: %s' % (col))
        df[col].fillna(0, inplace=True)
    print('Missing value imputation done.')

    del df['pymnt_plan']

    loan_to_income=df['annual_inc']/df['loan_amnt']
    df['loan_to_income']=loan_to_income
    df['neg_state'] = df['acc_now_delinq'] + (df['total_rec_late_fee']/df['loan_amnt']) + (df['recoveries']/df['loan_amnt']) + (df['collection_recovery_fee']/df['loan_amnt']) + (df['collections_12_mths_ex_med']/df['loan_amnt'])
    df.loc[df['neg_state'] > 0, 'neg_state'] = 1
    df['int_paid'] = df['total_rec_int'] + df['total_rec_late_fee']
    df['total_repayment_progress'] = ((df['last_week_pay']/(df['term']/12*52+1))*100) + ((df['recoveries']/df['loan_amnt']) * 100)
    df['avl_lines'] = df['total_acc'] - df['open_acc']



    df['home_ownership'].replace(to_replace='MORTGAGE', value='2', regex=True, inplace=True)
    df['home_ownership'].replace(to_replace='RENT', value='3', regex=True, inplace=True)
    df['home_ownership'].replace(to_replace='OWN', value='1', regex=True, inplace=True)
    df['home_ownership'].replace(to_replace='OTHER', value='4', regex=True, inplace=True)
    df['home_ownership'].replace(to_replace='NONE', value='5', regex=True, inplace=True)
    df['home_ownership'].replace(to_replace='ANY', value='6', regex=True, inplace=True)

    df['verification_status'].replace(to_replace='Source Verified', value='1', regex=True, inplace=True)
    df['verification_status'].replace(to_replace='Not Verified', value='3', regex=True, inplace=True)
    df['verification_status'].replace(to_replace='Verified', value='2', regex=True, inplace=True)


    from sklearn.model_selection import train_test_split
    feature_col_names=['term','int_rate', 'sub_grade','home_ownership','verification_status','dti','delinq_2yrs','inq_last_6mths','pub_rec','initial_list_status','recoveries',
                      'collections_12_mths_ex_med','application_type','acc_now_delinq','neg_state']
    predected_class_names=['loan_status']

    X=df[feature_col_names].values #predector feature column
    y=df[predected_class_names].values #predicted class(1=true 0=false) column
    split_test_size=0.30

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=split_test_size, random_state=42) #test_size=0.3 is 30% and 42 is seed for random nos generator


    #Logistic Regression again with new c
    from sklearn.linear_model import LogisticRegression

    lr_model =LogisticRegression(class_weight='balanced',C=0.7,random_state=42)
    lr_model.fit(X_train,y_train.ravel())
    lr_predict_test=lr_model.predict(X_test)

    #training
    print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test,lr_predict_test)))

    #Logistic Regression (CV cross validation)
    from sklearn.linear_model import LogisticRegressionCV
    lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced")  # set number of jobs to -1 which uses all cores to parallelize\n",
    lr_cv_model.fit(X_train, y_train.ravel())

    lr_cv_predict_test = lr_cv_model.predict(X_test)
    #training metrics
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test)))

    accu=metrics.accuracy_score(y_test, lr_cv_predict_test)
    accu=accu*100
    accu=round(accu,2)
    accu=str(accu)+"%"
    t2.delete("1.0", END)
    t2.insert(END,accu)

    y=[a,b,x,d,e,f,g,h,i,j,k,l,m,n,o]
    #y=list()
    #for i in range(0,len(lst)):
        #x=float(input(lst[i]))
        #y.append(x)
    y=y[0:15]
    status_x=[y]
    status_y=lr_cv_model.predict(status_x)
    #print()
    if status_y[0] == 1:
        
        pr="Loan Defaulter"
    else:
        pr="Not a loan Defaulter"

    t1.delete("1.0", END)
    t1.insert(END,pr)





e1=Label(window,text="Loan Term")
e1.grid(row=0,column=0)
e1_value=StringVar()
e11=Entry(window,textvariable=e1_value)
e11.grid(row=0,column=1)


e2=Label(window,text="Interest rate")
e2.grid(row=0,column=2)
e2_value=StringVar()
e22=Entry(window,textvariable=e2_value)
e22.grid(row=0,column=3)


e3=Label(window,text="Sub grade")
e3.grid(row=0,column=4)
e3_value=StringVar()
e33=Entry(window,textvariable=e3_value)
e33.grid(row=0,column=5)


e4=Label(window,text="home ownership")
e4.grid(row=1,column=0)
e4_value=StringVar()
e44=Entry(window,textvariable=e4_value)
e44.grid(row=1,column=1)


e5=Label(window,text="verification status")
e5.grid(row=1,column=2)
e5_value=StringVar()
e55=Entry(window,textvariable=e5_value)
e55.grid(row=1,column=3)


e6=Label(window,text="DTI")
e6.grid(row=1,column=4)
e6_value=StringVar()
e66=Entry(window,textvariable=e6_value)
e66.grid(row=1,column=5)


e7=Label(window,text="deliquency in 2 years")
e7.grid(row=2,column=0)
e7_value=StringVar()
e77=Entry(window,textvariable=e7_value)
e77.grid(row=2,column=1)


e8=Label(window,text="Inuqiries in last 6 months ")
e8.grid(row=2,column=2)
e8_value=StringVar()
e88=Entry(window,textvariable=e8_value)
e88.grid(row=2,column=3)


e9=Label(window,text="public Records")
e9.grid(row=2,column=4)
e9_value=StringVar()
e99=Entry(window,textvariable=e9_value)
e99.grid(row=2,column=5)

e10=Label(window,text="initial list status")
e10.grid(row=3,column=0)
e10_value=StringVar()
e100=Entry(window,textvariable=e10_value)
e100.grid(row=3,column=1)

e11=Label(window,text="recoveries")
e11.grid(row=3,column=2)
e11_value=StringVar()
e111=Entry(window,textvariable=e11_value)
e111.grid(row=3,column=3)

e12=Label(window,text="collections_12_mths_ex_med")
e12.grid(row=3,column=4)
e12_value=StringVar()
e122=Entry(window,textvariable=e12_value)
e122.grid(row=3,column=5)

e13=Label(window,text="Application Type")
e13.grid(row=4,column=0)
e13_value=StringVar()
e133=Entry(window,textvariable=e13_value)
e133.grid(row=4,column=1)

e14=Label(window,text="No of deliquent account")
e14.grid(row=4,column=2)
e14_value=StringVar()
e144=Entry(window,textvariable=e14_value)
e144.grid(row=4,column=3)

e15=Label(window,text="total_rec_late_fee")
e15.grid(row=4,column=4)
e15_value=StringVar()
e155=Entry(window,textvariable=e15_value)
e155.grid(row=4,column=5)

e16=Label(window,text="collection_recovery_fee")
e16.grid(row=5,column=0)
e16_value=StringVar()
e166=Entry(window,textvariable=e16_value)
e166.grid(row=5,column=1)


b1=Button(window,text='prediction',command=predict)
b1.grid(row=6,column=0)


t1=Text(window,height=1,width=25)
t1.grid(row=6,column=2)

e10=Label(window,text="Model accuracy")
e10.grid(row=6,column=4)

t2=Text(window,height=1,width=20)
t2.grid(row=6,column=5)

window.mainloop()
