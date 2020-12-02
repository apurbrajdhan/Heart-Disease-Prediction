__author__ = "home"
__date__ = "$4 April, 2020 10:46:35 AM$"

from flask import Flask
from flask import flash
from flask import redirect
from flask import render_template
from flask import request
from flask import session
import os
import pymysql
from werkzeug.utils import secure_filename
import os
print(os.listdir())
import warnings
warnings.filterwarnings('ignore')
from filtering import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz   
import tensorflow as tf
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense

UPLOAD_FOLDER = 'C:/Users/apurb/Desktop/web/dd/src/mitdb/'
configFileName = 'config.json'
ALLOWED_EXTENSIONS = set(['dat', 'atr', 'hea', 'xws'])

app = Flask(__name__)
app.secret_key = "123"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 * 1024
class Database:
    def __init__(self):
        host = "localhost"
        user = "root"
        password = ""
        db = "diseaseprediction"
        self.con = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.
                                   DictCursor)
        self.cur = self.con.cursor()
    def getpersonaldetails(self, username, password):
        strQuery = "SELECT COUNT(*) AS c, UserId FROM personaldetails WHERE Username = '" + username + "' AND Password = '" + password + "'"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def getprofiledetails(self, username):
        strQuery = "SELECT Firstname,Lastname,Phoneno,Address,UserId FROM personaldetails WHERE Username = '" + username + "' LIMIT 1"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        print(result)
        return result
    def geteducationdetails(self):
        strQuery = "SELECT * FROM education"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        return result
    def getdesiginationdetails(self):
        strQuery = "SELECT * FROM desigination"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        return result
    def getcountrydetails(self):
        strQuery = "SELECT * FROM country"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        return result
    def getstatedetails(self):
        strQuery = "SELECT * FROM state"
        self.cur.execute(strQuery)
        result = self.cur.fetchall()
        return result
    def insertprofiledetails(self, firstname, lastname, phone, email, address, education, desigination, country, state, username, password):
        print('insertprofiledetails::' + firstname)
        strQuery = "INSERT INTO personaldetails(Firstname, Lastname, Phoneno, Emailid, Address, EID, DID, CID, SID, Username, Password, Status) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        strQueryVal = (firstname, lastname, phone, email, address, education, desigination, country, state, username, password, 'Active')
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""    
    def insertvideodetails(self, userId, FileName, FileUrl, Status):
        print('insertvideodetails::' + FileName)
        strQuery = "INSERT INTO videofiles(UserId, FileName, FileUrl, Status, RecordedDate) values(%s, %s, %s, %s, now())"
        strQueryVal = (userId, FileName, FileUrl, Status)
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""
    def getvideodetails(self, userId, status):
        strQuery = "SELECT UploadId, FileName, FileUrl, Status, RecordedDate FROM videofiles WHERE Status = %s ORDER BY RecordedDate DESC LIMIT 2"
        self.cur.execute(strQuery, (status))
        result = self.cur.fetchall()
        print(result)
        return result
    def deletevideodetails(self, UploadId):
        print(UploadId)
        strQuery = "DELETE FROM videofiles WHERE UploadId = (%s) " 
        strQueryVal = (str(UploadId))
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""  
    def gettranhistory(self, userId):
        strQuery = "SELECT * FROM tranhistory WHERE UserId = %s ORDER BY RecordedDate DESC LIMIT 1"
        self.cur.execute(strQuery, (userId, ))
        result = self.cur.fetchall()
        print(result)
        return result
    def gettranhistorylist(self, userId):
        strQuery = "SELECT Vehicle_Time, Vehicle_Speed, RecordedDate FROM tranhistory WHERE UserId = %s ORDER BY RecordedDate DESC"
        self.cur.execute(strQuery, (userId, ))
        result = self.cur.fetchall()
        print(result)
        return result
    def inserttranhistory(self, userId, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, status):
        print(type(str(userId)))
        strQuery = "INSERT INTO tranhistory(UserId, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, Status, RecordedDate) values('"+str(userId)+"', '"+age+"', '"+sex+"', '"+cp+"', '"+trestbps+"', '"+chol+"', '"+fbs+"', '"+restecg+"', '"+thalach+"', '"+exang+"', '"+oldpeak+"', '"+slope+"', '"+ca+"', '"+thal+"', '"+status+"', now())"
        print(strQuery)
        self.cur.execute(strQuery)
        self.con.commit()
        return ""
    def updatevideodetails(self, userId, Status):
        print('updatevideodetails::' + Status)
        strQuery = "UPDATE videofiles SET Status = %s"
        strQueryVal = (Status)
        self.cur.execute(strQuery, strQueryVal)
        self.con.commit()
        return ""
        
@app.route('/', methods=['GET'])
def loadindexpage():
    return render_template('index.html')

@app.route('/codeindex', methods=['POST'])
def codeindex():
    username = request.form['username']
    password = request.form['password']
    
    print('username:' + username)
    print('password:' + password)
    
    try:
        if username is not "" and password is not "": 
            def db_query():
                db = Database()
                emps = db.getpersonaldetails(username, password)       
                return emps
            res = db_query()
            
            for row in res:
                print(row['c'])
                count = row['c']
                
                if count >= 1:      
                    session['x'] = username;
                    session['UID'] = row['UserId'];
                    def db_query():
                        db = Database()
                        emps = db.getprofiledetails(username)       
                        return emps
                    profile_res = db_query()
                    return render_template('profile.html', sessionValue=session['x'], result=profile_res, content_type='application/json')
                else:
                    flash ('Incorrect Username or Password.')
                    return render_template('index.html')
        else:
            flash ('Please fill all mandatory fields.')
            return render_template('index.html')
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('index.html')
        
    return render_template('index.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET'])
def signin():
    def db_query():
        db = Database()
        emps = db.geteducationdetails()       
        return emps
    edu_res = db_query()
    
    def db_query():
        db = Database()
        emps = db.getdesiginationdetails()       
        return emps
    designation_res = db_query()
    
    def db_query():
        db = Database()
        emps = db.getcountrydetails()       
        return emps
    country_res = db_query()
    
    def db_query():
        db = Database()
        emps = db.getstatedetails()       
        return emps
    state_res = db_query()
    
    return render_template('signin.html', eduresult=edu_res, desresult=designation_res, couresult=country_res, staresult=state_res, content_type='application/json')

@app.route('/signout', methods=['GET'])
def signout():    
    return render_template('signout.html')

@app.route('/home', methods=['GET'])
def home():
    def db_query():
        db = Database()
        emps = db.getprofiledetails(session['x'])       
        return emps
    profile_res = db_query()
    return render_template('profile.html', sessionValue=session['x'], result=profile_res, content_type='application/json')

@app.route('/logout', methods=['GET'])
def logout():
    del session['x']
    return render_template('index.html')

@app.route('/codesignin', methods=['POST'])
def codesignin():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    phone = request.form['phone']
    email = request.form['email']
    address = request.form['address']
    education = request.form['education']
    desigination = request.form['desigination']
    country = request.form['country']
    state = request.form['state']
    username = request.form['username']
    password = request.form['password']
    
    print('codesignin')
    
    try:
        if firstname is not "" and lastname is not ""  and phone is not "" and email is not "" and address is not "" and education is not "" and desigination is not "" and country is not "" and state is not "" and username is not "" and password is not "": 
            def db_query():
                db = Database()
                emps = db.getpersonaldetails(username, password)       
                return emps
            res = db_query()

            for row in res:
                print(row['c'])
                count = row['c']

                if count >= 1:      
                    flash ('Entered details already exists.')
                    return render_template('signin.html')
                else:
                    def db_query():
                        db = Database()
                        emps = db.insertprofiledetails(firstname, lastname, phone, email, address, education, desigination, country, state, username, password)       
                        return emps
                res = db_query()
                flash ('Dear Customer, Your registration has been done successfully.')
                return render_template('index.html')
        else:
            def db_query():
                db = Database()
                emps = db.geteducationdetails()       
                return emps
            edu_res = db_query()

            def db_query():
                db = Database()
                emps = db.getdesiginationdetails()       
                return emps
            designation_res = db_query()

            def db_query():
                db = Database()
                emps = db.getcountrydetails()       
                return emps
            country_res = db_query()

            def db_query():
                db = Database()
                emps = db.getstatedetails()       
                return emps
            state_res = db_query()
            
            flash ('Please fill all mandatory fields.')
            return render_template('signin.html', eduresult=edu_res, desresult=designation_res, couresult=country_res, staresult=state_res, content_type='application/json')
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('index.html')
    
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
def upload():       
    return render_template('upload.html', sessionValue=session['x'])

@app.route('/codeupload', methods=['POST'])
def codeupload():
    
    if 'filepath' not in request.files:
        flash('No file part')
        return render_template('upload.html', sessionValue=session['x'])
    file = request.files['filepath']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
       # return redirect('viewupload.html')
        def db_query():
            db = Database()
            emps = db.getprofiledetails(session['x'])       
            return emps
        res = db_query()
        
        for row in res:
            print(row['UserId'])
            userId = row['UserId']
        
        Status  = "";
        print(filename);
        xc=filename
        #xc = [i.strip().split() for i in open(app.config['UPLOAD_FOLDER'] + "/" + filename).readlines()]
        rec_index = "119"
        sampling_freq = 1000
        cutoff_freq = 100
        low_cutoff_freq = 200
        high_cutoff_freq = 300
        sampfrom = 0
        sampto = 2000

        rec_index = xc[len(xc)-7:len(xc)-4]
        sampfrom = 0
        sampto = 1000

        import wfdb

        wfdb.show_ann_labels()
        
        '''
        for rec in DS1:
            annotations = wfdb.rdann("mitdb/" + rec, "atr")
            annotations.get_contained_labels()
            #print(annotations.contained_labels)
            #print()
        '''
        sampfrom = 0
        sampto = 2000

        import cv2

        # Load an color image in grayscale
        cv2.waitKey(0)

        #rec_index = "119"
        sampfrom = 0
        sampto = 1000

        def autocorr (x, mode="full"):
            y = np.convolve(x, x, mode)
            return y[int(y.size/2) :] # / y[int(y.size/2)]    # Normalized

        import wfdb
        from statsmodels.tsa.stattools import levinson_durbin
        from collections import OrderedDict
        from filtering import butter_filter

        def extract_features (record_path, length_qrs, length_stt, ar_order_qrs, ar_order_stt, sampfrom=0, sampto=-1, use_filter=True):

            """
            A list holding tuples with values 'N' or 'VEB', and the length in samples of each corresponding QRS
            and ST/T complexes, plus the length in samples of pre- and post-RR
            """
            print(record_path)
            qrs_stt_rr_list = list()
            sampto = 10000
            #print(sampto)

            if sampto < 0:
                raw_signal, _= wfdb.rdsamp(record_path, channels=[0], sampfrom=sampfrom, sampto="end")
                annotations = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=None)
            else:
                raw_signal,_ = wfdb.rdsamp(record_path, channels=[0], sampfrom=sampfrom, sampto=sampto)
                annotations = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=sampto)

            raw_signal = raw_signal.reshape(-1)

            # Filtering
            if use_filter:
                filter_1 = butter_filter(raw_signal, filter_type="highpass", order=3, cutoff_freqs=[1], sampling_freq=annotations.fs)
                filter_2 = butter_filter(filter_1, filter_type="bandstop", order=3, cutoff_freqs=[58, 62], sampling_freq=annotations.fs)
                signal = butter_filter(filter_2, filter_type="lowpass", order=4, cutoff_freqs=[25], sampling_freq=annotations.fs)
            else:
                signal = raw_signal

            annotation2sample = list(zip(annotations.symbol, annotations.sample))

            for idx, annot in enumerate(annotation2sample):
                beat_type       = annot[0]    # "N", "V", ... etc.
                r_peak_pos      = annot[1]    # The R peak position
                pulse_start_pos = r_peak_pos - int(length_qrs / 2) + 1    # The sample postion of pulse start (start of QRS)

                # We treat only Normal, VEB, and SVEB signals (See the paper)
                print(beat_type)
                if beat_type == "N" or beat_type == "S" or beat_type == "V" or beat_type == "L" or beat_type == "R":
                    qrs = signal[pulse_start_pos : pulse_start_pos + length_qrs]
                    stt = signal[pulse_start_pos + length_qrs + 1 : pulse_start_pos + length_qrs + length_stt]
                    #print(qrs.size)
                    if qrs.size > 0:
                        _, qrs_arcoeffs, _, _, _ = levinson_durbin(qrs, nlags=ar_order_qrs, isacov=False)
                    else:
                        qrs_arcoeffs = None
                    #print(stt.size)  
                    if stt.size > 0:
                        #print(stt.shape)
                        _, stt_arcoeffs, _, _, _ = levinson_durbin(stt, nlags=ar_order_stt, isacov=False)
                    else:
                        stt_arcoeffs = None

                    pre_rr_length  = annotation2sample[idx][1] - annotation2sample[idx - 1][1] if idx > 0 else None
                    post_rr_length = annotation2sample[idx + 1][1] - annotation2sample[idx][1] if idx + 1 < annotations.ann_len  else None
                    _type = 1 if beat_type == "V" else 0

                    """
                    beat_dict = OrderedDict([("record", record_path.rsplit(sep="/", maxsplit=1)[-1]), ("type", _type),
                                             ("QRS", qrs), ("ST/T", stt),
                                             ("QRS_ar_coeffs", qrs_arcoeffs), ("ST/T_ar_coeffs", stt_arcoeffs),
                                             ("pre-RR", pre_rr_length), ("post-RR", post_rr_length)])
                    """
                    beat_list = list()
                    beat_list = [("record", record_path.rsplit(sep="/", maxsplit=1)[-1]), ("type", _type), 
                                 # ("QRS", qrs), ("ST/T", stt),
                                 # ("QRS_ar_coeffs", qrs_arcoeffs), ("ST/T_ar_coeffs", stt_arcoeffs),
                                 ("pre-RR", pre_rr_length), ("post-RR", post_rr_length)
                                ]
                    #print(qrs_arcoeffs)
                    #print('gdlgh')
                    for idx, coeff in enumerate(qrs_arcoeffs):
                        beat_list.append(("qrs_ar{}".format(idx), coeff))
                    print(stt_arcoeffs)
                    for idx, coeff in enumerate(stt_arcoeffs):
                        beat_list.append(("stt_ar{}".format(idx), coeff))

                    beat_dict = OrderedDict(beat_list)

                    qrs_stt_rr_list.append(beat_dict)
            return qrs_stt_rr_list

        def series2arCoeffs (series):
            if series.size > 0:
                return np.concatenate(series.tolist()).reshape(series.size, -1)
            else:
                return None

        import numpy as np
        import pandas as pd

        # Check the paper for choosing the lengthes
        length_qrs = 40
        length_stt = 100

        lst = list()

        # Tweak the use_filter param
        lst.extend(extract_features("mitdb/" +rec_index, length_qrs, length_stt, ar_order_qrs=3, ar_order_stt=3, use_filter=True))

        df = pd.DataFrame(lst)
        df.dropna(inplace=True)

        y = df["type"].values

        from sklearn.preprocessing import scale
        from sklearn.model_selection import train_test_split
        X = df[df.columns[2:]].values
        X = scale(X)

        X_test=np.array(X)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        #from tensorflow.python.keras.models import Sequential,load_model

        #import keras.backend.tensorflow_backend as KTF
        #import tensorflow as tf
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True   
        #sess = tf.Session(config=config)

        #KTF.set_session(sess)

        #from keras.models import load_model
        model = tf.keras.models.load_model('Trained_model_real.h5')
        

        Y_pred_rnn = model.predict(X_test)

        if max(Y_pred_rnn)>.5:
            Status = 'Given patient data is Heart Disease Caution'
            print('Given patient data is Heart Disease Caution')    
        else:
             Status = 'Given patient data is Normal'
             print('Given patient data is Normal')
             
        def db_query():
            db = Database()
            emps = db.updatevideodetails(userId, Status)  
            return emps      
        res = db_query()
        flash('File successfully uploaded')
        
        def db_query():
            db = Database()
            emps = db.getprofiledetails(session['x'])       
            return emps
        res = db_query()

        for row in res:
            print(row['UserId'])
            userId = row['UserId']

        def db_query():
            db = Database()
            emps = db.getvideodetails(userId, Status)  
            return emps      
        video_res = db_query()

        return render_template('viewupload.html', sessionValue=session['x'], result=video_res, content_type='application/json')

    else:
        flash('Allowed file types are .jpg, .png')
        return redirect(request.url)
    
        def db_query():
            db = Database()
            emps = db.getprofiledetails(session['x'])       
            return emps
        res = db_query()

        for row in res:
            print(row['UserId'])
            userId = row['UserId']

        def db_query():
            db = Database()
            emps = db.getvideodetails(userId, Status)  
            return emps      
        video_res = db_query()

        return render_template('viewupload.html', sessionValue=session['x'], result=video_res, content_type='application/json')

@app.route('/viewupload', methods=['GET'])
def viewupload():    
    def db_query():
        db = Database()
        emps = db.getprofiledetails(session['x'])       
        return emps
    res = db_query()
    
    for row in res:
        print(row['UserId'])
        userId = row['UserId']

    def db_query():
        db = Database()
        emps = db.getvideodetails(userId)  
        return emps      
    video_res = db_query()
        
    return render_template('viewupload.html', sessionValue=session['x'], result=video_res, content_type='application/json')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html', sessionValue=session['x'])

@app.route('/realtime', methods=['GET'])
def realtime():
    return render_template('realtime.html', sessionValue=session['x'])

@app.route('/offline', methods=['GET'])
def offline():
    return render_template('offline.html', sessionValue=session['x'])

@app.route('/codeoffline', methods=['POST'])
def codeoffline():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    
    print('age:' + age)
    print('sex:' + sex)
    print('cp:' + cp)
    print('trestbps:' + trestbps)
    print('chol:' + chol)
    print('fbs:' + fbs)
    print('restecg:' + restecg)
    print('thalach:' + thalach)
    print('exang:' + exang)
    print('oldpeak:' + oldpeak)
    print('slope:' + slope)
    print('ca:' + ca)
    print('thal:' + thal)
    
    testdata = []
    status = '';
    
    try:
        if age is not "" and sex is not "" and cp is not "" and trestbps is not "" and chol is not "" and fbs is not "" and restecg is not "" and thalach is not "" and exang is not "" and oldpeak is not "" and slope is not "" and ca is not "" and thal is not "": 
        
            testdata.append(float(age))
            testdata.append(float(sex))
            testdata.append(float(cp))
            testdata.append(float(trestbps))
            testdata.append(float(chol))
            testdata.append(float(fbs))
            testdata.append(float(restecg))
            testdata.append(float(thalach))
            testdata.append(float(exang))
            testdata.append(float(oldpeak))
            testdata.append(float(slope))
            testdata.append(float(ca))
            testdata.append(float(thal))
            
            print('testdata:', testdata)
                     
            X_test = np.array(testdata)
            import tensorflow as tf
            from keras.models import load_model
            
            import keras.backend.tensorflow_backend as KTF
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True   
            sess = tf.Session(config=config)

            KTF.set_session(sess)
            X_test = np.reshape(X_test, (1, 13,1))

            model =tf.keras.models.load_model('Trained_model.h5')

            Y_pred_rnn = model.predict(X_test)

            if Y_pred_rnn>.5:

                status = 'Given patient data is Normal.';
                
                print('Given patient data is Normal.')

            else:

                 status = 'Given patient data is Heart Disease Caution.';
                 
                 print('Given patient data is Heart Disease Caution.')

            #def db_query():     
            db = Database()
            db.inserttranhistory(session['UID'], age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, status);
     
            def db_query():
                db = Database()
                emps = db.gettranhistory(session['UID'])  
                return emps      
            video_res = db_query()
        
            return render_template('viewtxnhistory.html', sessionValue=session['x'], result=video_res, content_type='application/json')
        
        else:
            flash ('Please fill all mandatory fields.')
            return render_template('offline.html', sessionValue=session['x'], content_type='application/json')
        
    except NameError:
        flash ('Due to technical problem, your request could not be processed.')
        return render_template('offline.html', sessionValue=session['x'], content_type='application/json')
        
    return render_template('offline.html', sessionValue=session['x'], content_type='application/json')
