from django.http import HttpResponseRedirect, HttpResponse
from django.http.response import StreamingHttpResponse
from django.contrib.auth.hashers import make_password, check_password
from django.shortcuts import render
from django.http import JsonResponse
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage, InvalidPage
import sqlite3
import json
import random
import time
import datetime
import pytz
import os
from math import *
from . import rules


def switch_time(time_stamp):
    '''
		功能：时间戳转化为北京时间
        输入：int类型，时间戳（毫秒级）
        输出：str类型，对应的北京时间
	'''
    mili_second = time_stamp % 1000
    new_time_stamp = floor(time_stamp / 1000)
    time_array = time.localtime(new_time_stamp) 
    time_real = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    time_real += '.'
    time_real += str(mili_second).rjust(3,'0') 
    #print(time_real)
    return time_real


def getCurTime():
    '''
    	功能：获取当前毫秒时间戳
        输出：int类型，时间戳（毫秒级）
    '''
    curtime = int(time.time() * 1000)
    return curtime


def show_index(request):    
    '''
    	功能：处理/index(主页)访问请求
    '''

    context = {}
    username = rules.judge_whether_loaded(request)
    if(username == None):
        #未登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    else:
        context['username'] = username
        return render(request, 'index.html', context)

def show_error(request):
    '''
    	功能：处理/error(错误)访问请求，如果未登录访问主页，历史记录，删除等会让你返回登录，已登录访问登录，注册会让你返回主页
    '''
    context = {}
    username = rules.judge_whether_loaded(request)
    if(username != None):
        #已经登录，报错
        context['logged'] = True
    else:
        #未登录
        context['logged'] = False
    return render(request, 'error.html', context)


def show_logout(request):
    '''
    	功能：处理/logout（注销）访问请求
    '''
    rules.logout(request)
    response = HttpResponseRedirect('/login')
    #response.set_cookie('session_id', None)
    return response

def show_logon(request):
    '''
    	功能：处理/logon（注册）访问请求
        如果注册成功，返回用户名
        注册失败，返回失败错误码
    '''
    context = {}
    show_data = {}
    username = rules.judge_whether_loaded(request)
    if(username != None):
        #已经登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    if request.method == 'POST':
        show_data = rules.logon(request)
        if "error" in show_data.keys():
            context["message"] = show_data["error"]
            context["result"] = "error"
        else:
            context["result"] = "注册成功！"
            context["message"] = "用户名为" + str(show_data["user"])   
    return render(request, 'logon.html', context)

def show_login(request):
    '''
    	功能：处理/login（登录）访问请求
        如果登录成功，重定向主页
        登陆失败，返回错误码
    '''
    context = {}
    show_data = {}
    username = rules.judge_whether_loaded(request)
    if(username != None):
        #已经登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    if request.method == 'POST':
        show_data = rules.login(request)
        if "error" in show_data.keys():
            context["message"] = show_data["error"]
            context["result"] = "error"    
            return render(request, 'login.html', context)
        else:
            response = HttpResponseRedirect('/index')  
            response.set_cookie('session_id', show_data["session"])
            return response
    else:
        return render(request, 'login.html', context)

        
def show_service(request):
    '''
    	功能：处理主页的查看图片功能
        将后端返回的base64图片转化为可以在前端img显示的src，然后返回
    '''
    response_content = rules.service(request)
    response_content["original"] = "data:image/jpeg;base64," + response_content["original"][ : ]
    response_content["result"] = "data:image/jpeg;base64," + response_content["result"][ : ]
    #print(response_content)
    return JsonResponse(response_content)

#x是历史记录，用于排序
def f(x):
    return int(x["time"])

def show_history(request):
    '''
    	功能：处理/history/的url，显示历史记录
        查找并显示全部/按时间查找的历史记录，并且分页显示
    '''
    context = {}
    username = rules.judge_whether_loaded(request)
    if(username == None):
        #未登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    else:
        context['username'] = username
    if request.method == "GET":
        #将前端输入的时间转化为时间戳传给后端，如果没输入时间，传None
        try:
            start_date = str(request.GET.get("start_date"))
            end_date = str(request.GET.get("end_date"))
            start_time = start_date[ : ] + " 00:00:00"
            end_time = end_date[ : ] + " 23:59:59"
            start = time.mktime(time.strptime(start_time, '%Y-%m-%d %H:%M:%S')) * 1000
            end = time.mktime(time.strptime(end_time, '%Y-%m-%d %H:%M:%S')) * 1000
            assert start and end
        except:
            start = None
            end = None
        #查询全部记录，按照时间降序排序
        get_list = rules.query_date(request, start, end)
        for item in get_list["list"]:
            item["time"] = int(item["time"])
        get_list["list"].sort(key = f, reverse = True)
        record_list = []
        #将返回数据处理成前端显示的数据
        for item in get_list["list"]:
            new_dict = {}
            new_dict["username"] = item["username"]
            new_dict["name"] = item["name"]
            new_dict["id"] = str(item["record_id"])
            if item["content"] == "No URL.":
                new_dict["type"] = "本地图片"
            else:
                new_dict["type"] = "网络下载"
            new_dict["time"] = switch_time(int(item["time"]))
            record_list.append(new_dict)
        #分页显示
        #将数据按照规定每页显示10条, 进行分割
        paginator = Paginator(record_list, 10)
        # 获取 url 后面的 page 参数的值, 首页不显示 page 参数, 默认值是 1 
        try:
            page = request.GET.get('page')
            records = paginator.page(page)
        except:
            #如果请求的页数不合法，返回第一页
            records = paginator.page(1)
    context['records'] = records
    return render(request, "history.html", context)

def show_details(request):
    '''
    	功能：处理/detail/的url，显示单个历史记录
        获取单个历史记录的数据和图片，并且显示
    '''
    username = rules.judge_whether_loaded(request)
    if(username == None):
        #未登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    record = rules.get(request)
    if "error" in record.keys():
        #未找到此记录，报错
        response = HttpResponseRedirect('/not_found')  
        return response
    #将返回的内容处理成前端显示的内容
    context = {}
    context["username"] = record["username"]
    context["id"] = record["record_id"]
    context["name"] = record["name"]
    context["time"] = switch_time(int(record["time"]))
    if record["content"] == "No URL.":
        context["type"] = "本地图片"
    else:
        context["type"] = "网络下载"
    context["original"] = "data:image/jpeg;base64," + record["base64_image"]
    context["result"] = "data:image/jpeg;base64," + record["base64_result"]
    return render(request, "detail.html", context)

def show_delete(request):
    '''
    	功能：处理/delete/的url，删除单个历史记录
    '''
    username = rules.judge_whether_loaded(request)
    if(username == None):
        #未登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    record = rules.delete(request)
    if "error" in record.keys():
        #未找到此记录，报错
        response = HttpResponseRedirect('/not_found')  
        return response
    response = HttpResponseRedirect('/history/')  
    return response

def show_not_found(request):
    '''
    	功能：处理/not_found访问请求，如果历史记录不存在/没权限访问/历史记录不合法，比如图片损坏，格式不符，无法下载，无法打开等，显示这个
    '''
    context={}
    return render(request, "not_found.html", context)
    
def delete_many(request):
    '''
    	功能：删除按时间选中的多个历史记录
    '''
    context = {}
    username = rules.judge_whether_loaded(request)
    if(username == None):
        #未登录，报错
        response = HttpResponseRedirect('/error')  
        return response
    else:
        context['username'] = username
        #将输入时间转化为时间戳
        try:
            start_date = str(request.POST.get("start_date"))
            end_date = str(request.POST.get("end_date"))
            start_time = start_date[ : ] + " 00:00:00"
            end_time = end_date[ : ] + " 23:59:59"
            start = time.mktime(time.strptime(start_time, '%Y-%m-%d %H:%M:%S')) * 1000
            end = time.mktime(time.strptime(end_time, '%Y-%m-%d %H:%M:%S')) * 1000
            assert start and end
        except:
            start = None
            end = None
        #查询得到所有数据
        get_list = rules.query_date(request, start, end)
        delete_list = []
        #挨个删除数据库数据和文件
        for item in get_list["list"]:
            rules.delete_record(item["username"], int(item["record_id"]))
            try:
                os.remove(os.path.join("database", item["username"], str(item["record_id"]) + ".jpeg"))
                os.remove(os.path.join("database", item["username"], str(item["record_id"]) + "_result.jpeg"))
            except:
                pass
    return render(request, "history.html", context)
