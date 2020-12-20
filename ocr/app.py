import numpy as np
from flask import Flask, request, jsonify, render_template, Response, send_file, abort
import os
import cv2
# from keras.models import model_from_json
from tensorflow.keras.models import load_model


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# start flask
app = Flask(__name__)


# render default webpage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	image = request.files["images"]
	image_name = image.filename
	image.save(os.path.join(os.getcwd(), image_name))
	im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
	# Invert
	im = 255 - im
	# Calculate horizontal projection
	proj = np.sum(im,1)
	# Create output image same height as text, 500 px wide
	m = np.max(proj)
	w = 500
	result = np.zeros((proj.shape[0],500))
	proj=(proj)*w/m
	c = max(proj)
	r = np.where(proj==c)
	upper = im[0:r[0][0]-10,:]
	kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	upper = cv2.morphologyEx(upper, cv2.MORPH_DILATE, kernel3)
	print('upper shape',upper.shape)
	cont_upper, _  = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print('len of upper cont',len(cont_upper))
	lower = im[r[0][0]+5:,:]
	cont_lower, _  = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	u =[]
	if len(cont_upper) !=0:
		for c in sort_contours(cont_upper):
			(x, y, w, h) = cv2.boundingRect(c)
			u.append([x,x+w,y,y+h])
			curr_num = upper[y:y+h,x:x+w]
	l =[]
	for c in sort_contours(cont_lower):
		(x, y, w, h) = cv2.boundingRect(c)
		l.append([x,x+w+3,r[0][0]+y,r[0][0]+y+h])
		curr_num = lower[y:y+h,x:x+w+3]

	u_d = u.copy()
	new_c = []
	for i in range(len(l)):
		if len(u_d) !=0:
			if int(l[i][0]) in range(int(u_d[0][0]),int(u_d[0][1])):
				new_c.append([min(u_d[0][0],l[i][0]),max(u_d[0][1],l[i][1]),u_d[0][2],l[i][3]])
				u_d.pop(0)
			else:
				new_c.append([l[i][0],l[i][1],l[i][2]-7,l[i][3]])
		else:
			new_c.append([l[i][0],l[i][1],l[i][2]-7,l[i][3]])
	charactres = []
	output_chars=[]
	for i in range(len(new_c)):
		curr_num = im[new_c[i][2]:new_c[i][3],new_c[i][0]:new_c[i][1]]
		charactres.append(curr_num)
		loaded_model=load_model('cnn.hdf5')
		characters = '०,१,२,३,४,५,६,७,८,९,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ'
		characters = characters.split(',')
		resized = cv2.resize(curr_num, (32,32), interpolation = cv2.INTER_AREA)
		x = np.asarray(resized, dtype = np.float32).reshape(1, 32, 32, 1) / 255 
		output = loaded_model.predict(x)
		output = output.reshape(46)
		predicted = np.argmax(output)
		devanagari_label = characters[predicted]
		success = output[predicted] * 100
		print("the label is",devanagari_label)
		output_chars.append(devanagari_label)
	output = ''.join(output_chars)
	return render_template('index.html', prediction_text='{}'.format(output))
	# return jsonify(output)




if __name__ == "__main__":
    app.run(debug=True)
