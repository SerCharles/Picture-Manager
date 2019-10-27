from django.http import HttpResponse
from django.http import JsonResponse
from django.http import FileResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import numpy as np
from PIL import Image
from io import BytesIO
import random
import sqlite3
import sys
import os
import time
import configparser
import urllib.request
import base64
import json
import imageio
import scipy.misc as misc
import traceback


admin = ""
has_admin = False

# load inference networks for: segmentation and depth
import Kitti.depth as depth_net
depth_net.setup("Kitti/DATA/model_sn")
import Kitti.demo as seg_net
seg_net.setup(None)

def initialize_db():
	'''
	Initialize the database.
	Logon for administrator.
	'''
	global has_admin
	global admin

	success = True
	if not os.path.exists("./database"):
		try:
			os.mkdir("./database")
		except:
			sys.stderr.write("Error: database folder can not be created.")
			success = False

	if success:
		try:
			user_passwd = sqlite3.connect('./database/user_passwd.db')
			session_user = sqlite3.connect('./database/session_user.db')
		except:
			sys.stderr.write("create database fail.")
			success = False

	try:
		user_passwd_cursor = user_passwd.cursor()
		user_passwd_cursor.execute(
			'''
			CREATE TABLE USER_PASSWD
			(USERNAME 	CHAR(32)	PRIMARY	KEY 	NOT NULL 	UNIQUE,
			 PASSWORD 	CHAR(32)					NOT NULL					
			);
			'''
			)
		user_passwd.commit()
		user_passwd.close()

		session_user_cursor = session_user.cursor()
		session_user_cursor.execute(
			'''
			CREATE TABLE SESSION_USER
			(SESSION_ID 	INT 	PRIMARY KEY 	NOT NULL 	UNIQUE,
			 USERNAME 		CHAR(32)				NOT NULL,
			 LOGIN 			CHAR(5)  				NOT NULL
			);
			'''
			)
		session_user.commit()
		session_user.close()
	except:
		success = True

	# add administrator account if enabled
	parser = configparser.ConfigParser()
	parser.read("config.ini")
	try:
		admin = parser["administrator"]
		assert admin["username"] and admin["password"]
		res = new_user(admin["username"], admin["password"])
		has_admin = True if res in ["", "user exists"] else False

	except:
		has_admin = False

	if has_admin:
		admin = admin["username"]
	else:
		admin = ""
	if has_admin:
		print("Has administrator.")
	else:
		print("No administrator.")

	return success

def check_user(username, password):
	'''
		check :
			1. if username exist. 
			2. if username correspond with password.
		output:
			result: ["no such a user", "password is wrong", "", "database lost"]
	'''
	result = ""
	if not result:
		try:
			user_passwd = sqlite3.connect('./database/user_passwd.db')
			c = user_passwd.cursor()
		except:
			result = "database lost"
	if not result:
		info = c.execute('''
				SELECT PASSWORD FROM USER_PASSWD 
				WHERE USERNAME ="''' + username + '";'
				)
		
		pas = None
		for x in info:
			pas = x[0]
		user_passwd.close()
		if not pas:
			result = "no such a user"
		elif pas != password:
			result = "password is wrong"
		else:
			result = ""
	return result

def return_all_user():
	'''
		return all the users in the database.
	'''
	result = ""
	try:
		user_passwd = sqlite3.connect('./database/user_passwd.db')
		c = user_passwd.cursor()
	except:
		result = "database lost"
	if not result:
		info = c.execute('''
				SELECT USERNAME FROM USER_PASSWD;'''
				)
		info = [ele[0] for ele in info]
		user_passwd.close()

	return info

def new_user(username, password):
	'''
		if username do not exist, create one.

		output:
			result: ["user exists", "", "illegal input"]
	'''
	result = ""
	if not result:
		try:
			user_passwd = sqlite3.connect('./database/user_passwd.db')
			c = user_passwd.cursor()
		except:
			result = "database lost"
	if not result:
		if len(username) > 32 or len(password) > 32:
			result = "illegal input"
	if not result:
		try:
			c.execute("INSERT INTO USER_PASSWD (USERNAME, PASSWORD)\
				VALUES ('" + username + "', '" + password + "')")
			user_passwd.commit()
			user_passwd.close()
		except:
			result = "user exists"
	if not result:
		user = sqlite3.connect('./database/' + username + '.db')
		cursor = user.cursor()
		cursor.execute(
			'''
			CREATE TABLE USER
			(NAME 			CHAR(64) 				NOT NULL,
			 TIME_			INT 					NOT NULL,	
			 CONTENT 	CHAR(2000)					NOT NULL
			);
			'''
			)
		user.commit()
		user.close()
	return result

def login_user(username, session_id):
	'''
		login

		output:
			result: ["has logged in", "", "session user database lost"]
	'''
	result = ""
	if not result:
		try:
			session_user = sqlite3.connect('./database/session_user.db')
			c = session_user.cursor()
		except:
			result = "session user database lost"
	if not result:
		info = c.execute('''
				SELECT LOGIN FROM SESSION_USER 
				WHERE SESSION_ID ="''' + str(session_id) + '" AND USERNAME = "' + username + '";'
				)
		select = None
		for x in info:
			select = x
		if select and select[0] == "TRUE":
			result = "has logged in"
		else:
			try:
				if not select:
					c.execute("INSERT INTO SESSION_USER (SESSION_ID, USERNAME, LOGIN)\
						VALUES (" + str(session_id) + ", '" + username + "', 'TRUE')")
				else:
					c.execute("UPDATE SESSION_USER SET LOGIN = 'TRUE'" + 
						" WHERE SESSION_ID = '" + str(session_id) + "' AND USERNAME = '" + username + "';")
				session_user.commit()
				result = ""
			except sqlite3.IntegrityError as e:
				result = "has logged in"
		session_user.close()
	return result

def logout_user(session_id):
	'''
		output: 
			result: ["", "session user database lost"]
	'''
	result = ""
	if not result:
		try:
			session_user = sqlite3.connect('./database/session_user.db')
			c = session_user.cursor()
		except:
			result = "session user database lost"
	if not result:
		c.execute("DELETE FROM SESSION_USER WHERE SESSION_ID = '" + str(session_id) + "';")
		session_user.commit()
		result = ""
		session_user.close()
	return result

def check_login(session_id):
	'''
		check if a session_id is logged in.

		output:
			username (null if please login)
			result: ["", "session user database lost", "please login"]
	'''
	username = ""
	result = ""
	if not result:
		try:
			session_user = sqlite3.connect('./database/session_user.db')
			c = session_user.cursor()
		except:
			result = "session user database lost"
	if not result:
		info = c.execute('''
				SELECT USERNAME FROM SESSION_USER 
				WHERE SESSION_ID ="''' + str(session_id) + '" AND LOGIN = "TRUE";'
				)
		info = [x for x in info]
		if info:
			result = ""
			username = info[0][0]
		else:
			result = "please login"
		session_user.close()
	return username, result

def check_session_id(session_id):
	'''
		check if a session_id exists.

		output:
			result: ["exist", "session user database lost", "not exist"]
	'''
	result = ""
	if not result:
		try:
			session_user = sqlite3.connect('./database/session_user.db')
			c = session_user.cursor()
		except:
			result = "session user database lost"
	if not result:
		info = c.execute('''
				SELECT USERNAME FROM SESSION_USER 
				WHERE SESSION_ID ="''' + str(session_id) + '";'
				)
		info = [x for x in info]
		if info:
			result = "exist"
		else:
			result = "not exist"
		session_user.close()
	return result

def insert_record(username, name, time, content):
	'''
		insert a record to username's databse

		output:
			result: ["", "user database not found"]
			record_id
	'''
	result = ""
	record_id = -1
	if not result:
		try:
			user = sqlite3.connect('./database/' + username + '.db')
			c = user.cursor()
		except:
			result = "user database not found"
	if not result:
		c.execute("INSERT INTO USER (NAME, TIME_, CONTENT)\
				VALUES ('" + name + "', " + str(time) + ", '" + content + "')")
		record_id = c.execute("SELECT last_insert_rowid();")
		record_id = [x for x in record_id]
		user.commit()
		result = ""
	return result, record_id[0][0]

def delete_record(username, record_id):
	'''
		delete a record

		output:
			result: ["", "user database not found", "unknown record"]
	'''
	result = ""
	if not result:
		try:
			user = sqlite3.connect('./database/' + username + '.db')
			c = user.cursor()
		except:
			result = "user database not found"
	if not result:
		cursor = c.execute("SELECT NAME, TIME_, CONTENT FROM USER WHERE ROWID = " + str(record_id) + ";")
		cursor = [x for x in cursor]
		if cursor:
			c.execute("DELETE FROM USER WHERE ROWID = " + str(record_id) + ";")
			user.commit()
			result = ""
	return result

def get_record(username, record_id):
	'''
		output:
			record:{"record_id": 124, "name": "hello", "content": "hello world", "time": "2019-08-31 12:00:00.900"}
			result: ["unknown record", "", "user database not found"]
	'''
	result = ""
	record = {}
	if not result:
		try:
			user = sqlite3.connect('./database/' + username + '.db')
			c = user.cursor()
		except:
			result = "user database not found"
	if not result:
		cursor = c.execute("SELECT NAME, TIME_, CONTENT FROM USER WHERE ROWID = " + str(record_id) + ";")
		cursor = [x for x in cursor]
		if len(cursor):
			record["record_id"] = record_id
			record["name"] = cursor[0][0]
			record["content"] = cursor[0][2]
			record["time"] = cursor[0][1]
		else:
			result = "unknown record"
	return record, result

def query_record(username, start, end):
	'''
		return all the records of username with time ranging from start to end.
	'''
	result = ""
	if not result:
		try:
			user = sqlite3.connect('./database/' + username + '.db')
			c = user.cursor()
		except:
			result = "user database not found"
	if not result:
		if start == None and end == None:
			cursor = c.execute("SELECT ROWID, * FROM USER;")
		else:
			cursor = c.execute("SELECT ROWID, * FROM USER WHERE TIME_ < " + str(end) + " AND TIME_ > " + str(start) + ";")
		list_ = [{"username": username, "record_id": x[0], "name": x[1], "time": x[2], "content": x[3]} for x in cursor]
	return {"list": list_}, result

def logon(request):
	'''
		logon for a user.
	'''
	username = ""
	passwd = ""
	result_code = ""
	try:
		username = request.POST["username"]
		passwd = request.POST["password"]
		assert username and passwd
	except:
		result_code = "invalid parameters."
	for item in str(username):
		# check whether a username is legal
		if item == '_' or (item >= '0' and item <= '9') or (item >= 'a' and item <= 'z') or (item >= 'A' and item <= 'Z'):
			continue
		else:
			result_code = "invalid parameters"  

	if not result_code:
		result_code = new_user(username, passwd)

	if not result_code:
		res = {"user": username}
	else:
		res = {"error": result_code}

	return res

def login(request):
	'''
		login for a user.
		generate, send and keep a cookie.
	'''
	username = ""
	passwd = ""
	result_code = ""
	status_code = 200
	try:
		session_id = request.COOKIES["session_id"]
	except:
		session_id = ""
	try:
		username = request.POST["username"]
		passwd = request.POST["password"]
		assert username
	except:
		result_code = "no such a user"
		status_code = 200

	if not result_code:
		result_code = check_user(username, passwd)

	if not result_code:
		if not session_id or not check_session_id(session_id):
			session_id = random.randint(1, 999999999999)
			while (check_session_id(session_id) == "exist"):
				session_id = random.randint(1, 999999999999)
		result_code = login_user(username, session_id)
	if not result_code:
		response = {"user": username,"session":session_id}
		return response
	else:
		return {"error": result_code}

def judge_whether_loaded(request):
	'''
		judge wheher a session is active.
	'''
	result_code = ""
	session_id = ""
	username = ""
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "no valid session"

	if not result_code:
		username, result_code = check_login(session_id)
	else:
		result_code == "no valid session"

	username = username if not result_code else None
	return username

def logout(request):
	'''
		logout for a user.
	'''
	result_code = ""
	session_id = ""
	username = ""
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "no valid session"

	if not result_code:
		username, result_code = check_login(session_id)
	if result_code != "please login":
		result_code = logout_user(session_id)
	else:
		result_code == "no valid session"

	if result_code == "":
		return {"user": username}
	else:
		return {"error": result_code}

def service(request):
	'''
		get a picture encoded in base64
		and respond with the processed picture.
	'''
	result_code = ""
	session_id = ""
	username = ""
	name = ""
	time = 0
	content = ""
	record_id = 0

	# get important imformation
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "please login"
	try:
		name = request.POST["name"]
		time = int(request.POST["time"])
		content = str(request.POST["content"])
		assert name and content and time > 0
	except:
		result_code = "invalid parameters"

	# check if the user is legal
	if not result_code:
		username, result_code = check_login(session_id)

	if not result_code:
		result_code, record_id = insert_record(username, name, time, content)

	# save uploaded pictures
	loc = os.path.join("database", username, str(record_id) + ".jpeg")
	result_loc = os.path.join("database", username, str(record_id) + "_result.jpeg")
	if not os.path.exists(os.path.dirname(loc)):
		os.mkdir(os.path.dirname(loc))
	if not result_code:
		if content != "No URL.":
			try:
				urllib.request.urlretrieve(content, filename = loc)
			except:
				result_code = "error downloading or writing image."
		else:
			try:
				data = request.FILES.get("source_image")
				default_storage.save(loc, ContentFile(data.read()))
			except:
				result_code = "error downloading or writing image."
			
	# encode the original image
	image = Image.open(loc)
	output_buffer = BytesIO()
	image.save(output_buffer, format = "JPEG")
	binary = output_buffer.getvalue()
	base64_data = base64.b64encode(binary)
	original = base64_data.decode()

	# process the image according to instruction
	image = np.array(Image.open(loc))	
	if name == "segmentation":
		image = seg_net.segmentation(image)
	else:
		image = depth_net.inference(image)
	imageio.imsave(result_loc, image)
	pil_image = Image.fromarray(image.astype('uint8')).convert('RGB')
	output_buffer = BytesIO()
	pil_image.save(output_buffer, format = "JPEG")
	binary = output_buffer.getvalue()
	base64_data = base64.b64encode(binary)
	result = base64_data.decode()
	
	# save the result
	imageio.imsave(result_loc, image)
	pil_image = Image.fromarray(image.astype('uint8')).convert('RGB')
	output_buffer = BytesIO()
	pil_image.save(output_buffer, format = "JPEG")
	binary = output_buffer.getvalue()
	base64_data = base64.b64encode(binary)
	result = base64_data.decode()

	# return
	if not result_code:
		return {"record_id": record_id, "result": result, "original": original}
	else:
		return {"error": result_code}

def delete(request):
	global admin
	'''
		delete the requested records
	'''
	result_code = ""
	session_id = ""
	username = ""
	request_username = ''
	record_id = 0

	# check the request
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "please login"
	try:
		record_id = int(request.path.split('/')[-1])
	except:
		result_code = "invalid parameters"
	try:
		request_username = str(request.path.split('/')[-2])
	except:
		result_code = "invalid parameters"

	if not result_code:
		username, result_code = check_login(session_id)
	# check if the request is legal
	if username != admin and username != request_username:
		result_code = "unknown record"
	# search and delete the records
	if not result_code:
		record, result_code = get_record(request_username, record_id)
	if not result_code:
		result_code = delete_record(request_username, record_id, )
	
	# also delete images
	if not result_code:
		try:
			os.remove(os.path.join("database", request_username, str(record["record_id"]) + ".jpeg"))
			os.remove(os.path.join("database", request_username, str(record["record_id"]) + "_result.jpeg"))
		except:
			pass
	if not result_code:
		return {"record_id": record_id}
	else:
		return {"error": result_code}


def get(request):
	'''
		get a record, with source image and result
	'''
	global admin
	result_code = ""
	session_id = ""
	username = ""
	request_username = ""
	record_id = 0
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "please login"
	try:
		record_id = int(request.path.split('/')[-1])
	except:
		result_code = "unknown record"
	try:
		request_username = str(request.path.split('/')[-2])
	except:
		result_code = "invalid parameters"

	if not result_code:
		username, result_code = check_login(session_id)
	# check if the request is legal
	if username != admin and request_username != username:
		result_code = "unknown record"
	# search for the record
	if not result_code:
		record, result_code = get_record(request_username, record_id)

	if not result_code:
		loc = os.path.join("database", request_username, str(record["record_id"]) + ".jpeg")
		result_loc = os.path.join("database", request_username, str(record["record_id"]) + "_result.jpeg")
		try:
			with open(loc, "rb") as f:
				base64_data = base64.b64encode(f.read())
				s = base64_data.decode()
			with open(result_loc, "rb") as f:
				base64_data = base64.b64encode(f.read())
				r = base64_data.decode()
		except:
			result_code = "read image fail."
		else:
			record["base64_image"] = s
			record["base64_result"] = r

	if not result_code:
		record["username"] = request_username

	if not result_code:
		return record
	else:
		return {"error": result_code}


def query(request):
	global admin
	'''
		search for records with a time period.
		special authority granted for administrator.
	'''
	result_code = ""
	session_id = ""
	username = ""
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "please login"
	if not result_code:
		username, result_code = check_login(session_id)
	if not result_code:
		try:
			start = request.GET.get("start")
			end = request.GET.get("end")
			assert start and end
		except:
			start = None
			end = None
	if not result_code:
		list_, result_code = query_record(username, start, end)

	# administrator
	if username == admin:
		all_user = return_all_user()
		list_ = []
		for user in all_user:
			l_, result_code = query_record(user, start, end)
			list_ += l_["list"]
		list_ = {"list": list_}
	
	if not result_code:
		return list_
	else:
		return {"error": result_code}

def query_date(request, start, end):
	global admin
	'''
		search for records with a time period.
		special authority granted for administrator.
	'''
	result_code = ""
	session_id = ""
	username = ""
	try:
		session_id = request.COOKIES["session_id"]
	except:
		result_code = "please login"
	if not result_code:
		username, result_code = check_login(session_id)
	if not result_code:
		list_, result_code = query_record(username, start, end)
	
	if username == admin:
		all_user = return_all_user()
		list_ = []
		for user in all_user:
			l_, result_code = query_record(user, start, end)
			list_ += l_["list"]
		list_ = {"list": list_}
	
	if not result_code:
		return list_
	else:
		return {"error": result_code}